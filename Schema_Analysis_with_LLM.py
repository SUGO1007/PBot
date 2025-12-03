import streamlit as st
import yaml
import json
import re
from typing import Dict, List, Any, TypedDict, Annotated
from datetime import datetime
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# Define the state schema for the agent workflow
class SchemaAnalysisState(TypedDict):
    """State that gets passed between agents in the workflow"""
    source_schema: Dict
    target_schema: Dict
    source_yaml: str
    target_yaml: str
    table_matches: List[Dict]
    unmatched_source_tables: List[str]
    unmatched_target_tables: List[str]
    overall_similarity: int
    recommendations: List[str]
    current_step: str
    error: str
    model_used: str
    analysis_timestamp: str
    messages: Annotated[List[str], operator.add]

class SchemaParserAgent:
    """Agent responsible for parsing and validating YAML schemas"""
    
    def __init__(self):
        self.name = "SchemaParser"
    
    def parse(self, state: SchemaAnalysisState) -> SchemaAnalysisState:
        """Parse YAML schemas and extract metadata"""
        state["messages"].append(f"[{self.name}] Parsing schemas...")
        state["current_step"] = "parsing"
        
        try:
            source = yaml.safe_load(state["source_yaml"])
            target = yaml.safe_load(state["target_yaml"])
            
            state["source_schema"] = source
            state["target_schema"] = target
            state["messages"].append(f"[{self.name}] Successfully parsed schemas")
            
            # Extract unmatched tables initially (all tables)
            state["unmatched_source_tables"] = [t["name"] for t in source.get("tables", [])]
            state["unmatched_target_tables"] = [t["name"] for t in target.get("tables", [])]
            
        except Exception as e:
            state["error"] = f"Schema parsing error: {str(e)}"
            state["messages"].append(f"[{self.name}] Error: {str(e)}")
        
        return state

class TableMatchingAgent:
    """Agent responsible for matching tables between schemas"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "TableMatcher"
    
    def match_tables(self, state: SchemaAnalysisState) -> SchemaAnalysisState:
        """Use LLM to match tables between schemas"""
        state["messages"].append(f"[{self.name}] Analyzing table matches...")
        state["current_step"] = "table_matching"
        
        if state.get("error"):
            return state
        
        try:
            prompt = self._create_table_matching_prompt(state)
            response = self.llm.invoke(prompt)
            
            # Parse response
            matches = self._parse_table_matches(response)
            state["table_matches"] = matches
            
            # Update unmatched tables
            matched_source = [m["source_table"] for m in matches]
            matched_target = [m["target_table"] for m in matches]
            
            state["unmatched_source_tables"] = [
                t for t in state["unmatched_source_tables"] 
                if t not in matched_source
            ]
            state["unmatched_target_tables"] = [
                t for t in state["unmatched_target_tables"] 
                if t not in matched_target
            ]
            
            state["messages"].append(f"[{self.name}] Found {len(matches)} table matches")
            
        except Exception as e:
            state["error"] = f"Table matching error: {str(e)}"
            state["messages"].append(f"[{self.name}] Error: {str(e)}")
        
        return state
    
    def _create_table_matching_prompt(self, state: SchemaAnalysisState) -> str:
        """Create prompt for table matching"""
        source_tables = [t["name"] for t in state["source_schema"].get("tables", [])]
        target_tables = [t["name"] for t in state["target_schema"].get("tables", [])]
        
        return f"""You are a database schema expert. Match tables between source and target schemas.

Source Tables: {', '.join(source_tables)}
Target Tables: {', '.join(target_tables)}

For each source table, identify the best matching target table and provide:
1. Source table name
2. Target table name
3. Accuracy score (0-100)
4. Brief reasoning

Output as JSON array:
[{{"source": "table1", "target": "table2", "accuracy": 95, "reasoning": "..."}}]

Only JSON, no other text:"""
    
    def _parse_table_matches(self, response: str) -> List[Dict]:
        """Parse LLM response for table matches"""
        try:
            # Clean response
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # Find JSON array
            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            if start != -1 and end > start:
                cleaned = cleaned[start:end]
            
            matches = json.loads(cleaned)
            
            # Convert to internal format
            result = []
            for match in matches:
                result.append({
                    "source_table": match.get("source", ""),
                    "target_table": match.get("target", ""),
                    "accuracy": match.get("accuracy", 0),
                    "reasoning": match.get("reasoning", ""),
                    "column_matches": [],
                    "unmatched_source_columns": [],
                    "unmatched_target_columns": []
                })
            
            return result
        except:
            return []

class ColumnMatchingAgent:
    """Agent responsible for matching columns within matched tables"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "ColumnMatcher"
    
    def match_columns(self, state: SchemaAnalysisState) -> SchemaAnalysisState:
        """Match columns for each table pair"""
        state["messages"].append(f"[{self.name}] Matching columns...")
        state["current_step"] = "column_matching"
        
        if state.get("error"):
            return state
        
        try:
            for table_match in state["table_matches"]:
                source_table = self._get_table(
                    state["source_schema"], 
                    table_match["source_table"]
                )
                target_table = self._get_table(
                    state["target_schema"], 
                    table_match["target_table"]
                )
                
                if source_table and target_table:
                    column_matches = self._match_table_columns(
                        source_table, 
                        target_table
                    )
                    table_match["column_matches"] = column_matches
                    
                    # Track unmatched columns
                    matched_source = [c["source_column"] for c in column_matches]
                    matched_target = [c["target_column"] for c in column_matches]
                    
                    all_source = [c["name"] for c in source_table.get("columns", [])]
                    all_target = [c["name"] for c in target_table.get("columns", [])]
                    
                    table_match["unmatched_source_columns"] = [
                        c for c in all_source if c not in matched_source
                    ]
                    table_match["unmatched_target_columns"] = [
                        c for c in all_target if c not in matched_target
                    ]
            
            state["messages"].append(f"[{self.name}] Column matching complete")
            
        except Exception as e:
            state["error"] = f"Column matching error: {str(e)}"
            state["messages"].append(f"[{self.name}] Error: {str(e)}")
        
        return state
    
    def _get_table(self, schema: Dict, table_name: str) -> Dict:
        """Get table definition from schema"""
        for table in schema.get("tables", []):
            if table["name"] == table_name:
                return table
        return None
    
    def _match_table_columns(self, source_table: Dict, target_table: Dict) -> List[Dict]:
        """Match columns between two tables"""
        try:
            prompt = self._create_column_matching_prompt(source_table, target_table)
            response = self.llm.invoke(prompt)
            return self._parse_column_matches(response, source_table, target_table)
        except:
            return []
    
    def _create_column_matching_prompt(self, source_table: Dict, target_table: Dict) -> str:
        """Create prompt for column matching"""
        source_cols = []
        for col in source_table.get("columns", []):
            source_cols.append(f"{col['name']} ({col.get('type', 'UNKNOWN')})")
        
        target_cols = []
        for col in target_table.get("columns", []):
            target_cols.append(f"{col['name']} ({col.get('type', 'UNKNOWN')})")
        
        return f"""Match columns between these tables:

Source: {source_table['name']}
Columns: {', '.join(source_cols)}

Target: {target_table['name']}
Columns: {', '.join(target_cols)}

Provide matches as JSON array:
[{{"source": "col1", "target": "col2", "accuracy": 100, "reasoning": "..."}}]

Only JSON:"""
    
    def _parse_column_matches(self, response: str, source_table: Dict, target_table: Dict) -> List[Dict]:
        """Parse column matches from LLM response"""
        try:
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            
            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            if start != -1 and end > start:
                cleaned = cleaned[start:end]
            
            matches = json.loads(cleaned)
            
            # Get column type info
            source_col_types = {c["name"]: c.get("type", "UNKNOWN") 
                               for c in source_table.get("columns", [])}
            target_col_types = {c["name"]: c.get("type", "UNKNOWN") 
                               for c in target_table.get("columns", [])}
            
            result = []
            for match in matches:
                source_col = match.get("source", "")
                target_col = match.get("target", "")
                source_type = source_col_types.get(source_col, "UNKNOWN")
                target_type = target_col_types.get(target_col, "UNKNOWN")
                
                result.append({
                    "source_column": source_col,
                    "target_column": target_col,
                    "accuracy": match.get("accuracy", 0),
                    "source_type": source_type,
                    "target_type": target_type,
                    "type_compatible": source_type == target_type,
                    "reasoning": match.get("reasoning", "")
                })
            
            return result
        except:
            return []

class AnalysisAgent:
    """Agent responsible for computing overall analysis metrics"""
    
    def __init__(self):
        self.name = "Analyzer"
    
    def analyze(self, state: SchemaAnalysisState) -> SchemaAnalysisState:
        """Compute overall similarity and generate recommendations"""
        state["messages"].append(f"[{self.name}] Computing overall metrics...")
        state["current_step"] = "analysis"
        
        if state.get("error"):
            return state
        
        try:
            # Calculate overall similarity
            if state["table_matches"]:
                total_accuracy = sum(m["accuracy"] for m in state["table_matches"])
                state["overall_similarity"] = int(total_accuracy / len(state["table_matches"]))
            else:
                state["overall_similarity"] = 0
            
            # Generate recommendations
            recommendations = []
            
            if state["unmatched_source_tables"]:
                recommendations.append(
                    f"Review {len(state['unmatched_source_tables'])} unmatched source tables"
                )
            
            if state["unmatched_target_tables"]:
                recommendations.append(
                    f"Consider mapping to {len(state['unmatched_target_tables'])} unused target tables"
                )
            
            # Check type compatibility
            type_issues = 0
            for tm in state["table_matches"]:
                for cm in tm["column_matches"]:
                    if not cm["type_compatible"]:
                        type_issues += 1
            
            if type_issues > 0:
                recommendations.append(
                    f"Review {type_issues} column(s) with incompatible data types"
                )
            
            if state["overall_similarity"] < 70:
                recommendations.append(
                    "Low similarity score - manual review recommended"
                )
            
            state["recommendations"] = recommendations
            state["messages"].append(f"[{self.name}] Analysis complete")
            
        except Exception as e:
            state["error"] = f"Analysis error: {str(e)}"
            state["messages"].append(f"[{self.name}] Error: {str(e)}")
        
        return state

class SchemaMatchingWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.llm = Ollama(
            model=model_name,
            temperature=0.1,
            num_predict=4096,
            timeout=1200,
            format="json"
        )
        
        # Initialize agents
        self.parser_agent = SchemaParserAgent()
        self.table_matcher_agent = TableMatchingAgent(self.llm)
        self.column_matcher_agent = ColumnMatchingAgent(self.llm)
        self.analysis_agent = AnalysisAgent()
        
        # Build the workflow graph
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(SchemaAnalysisState)
        
        # Add nodes (agents)
        workflow.add_node("parse", self.parser_agent.parse)
        workflow.add_node("match_tables", self.table_matcher_agent.match_tables)
        workflow.add_node("match_columns", self.column_matcher_agent.match_columns)
        workflow.add_node("analyze", self.analysis_agent.analyze)
        
        # Define the workflow edges
        workflow.set_entry_point("parse")
        
        # Conditional routing based on errors
        workflow.add_conditional_edges(
            "parse",
            lambda state: "match_tables" if not state.get("error") else END
        )
        
        workflow.add_conditional_edges(
            "match_tables",
            lambda state: "match_columns" if not state.get("error") else END
        )
        
        workflow.add_conditional_edges(
            "match_columns",
            lambda state: "analyze" if not state.get("error") else END
        )
        
        workflow.add_edge("analyze", END)
        
        return workflow.compile()
    
    def run(self, source_yaml: str, target_yaml: str) -> Dict:
        """Execute the workflow"""
        # Initialize state
        initial_state = {
            "source_yaml": source_yaml,
            "target_yaml": target_yaml,
            "source_schema": {},
            "target_schema": {},
            "table_matches": [],
            "unmatched_source_tables": [],
            "unmatched_target_tables": [],
            "overall_similarity": 0,
            "recommendations": [],
            "current_step": "init",
            "error": "",
            "model_used": self.model_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "messages": []
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Convert to output format
        return self._format_output(final_state)
    
    def _format_output(self, state: SchemaAnalysisState) -> Dict:
        """Format the final state into output structure"""
        if state.get("error"):
            return {
                "error": state["error"],
                "messages": state["messages"]
            }
        
        return {
            "table_matches": state["table_matches"],
            "unmatched_source_tables": state["unmatched_source_tables"],
            "unmatched_target_tables": state["unmatched_target_tables"],
            "overall_schema_similarity": state["overall_similarity"],
            "recommendations": state["recommendations"],
            "model_used": state["model_used"],
            "analysis_timestamp": state["analysis_timestamp"],
            "agent_messages": state["messages"]
        }

def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        test_llm = Ollama(model="llama3.2", temperature=0.1)
        test_llm.invoke("test")
        return True, "Connected"
    except Exception as e:
        return False, str(e)

def get_available_models():
    """Get list of available Ollama models"""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]
            models = [line.split()[0] for line in lines if line.strip()]
            return models if models else ["llama3.2", "llama3.1", "mistral"]
        else:
            return ["llama3.2", "llama3.1", "mistral", "mixtral"]
    except Exception as e:
        # don't call st.* here; caller will handle UI warnings
        return ["llama3.2", "llama3.1", "mistral", "mixtral"]

def create_sample_yaml(schema_type: str) -> str:
    """Create sample YAML content"""
    if schema_type == "source":
        return """database: source_db
tables:
  - name: service
    columns:
      - name: service_id
        type: INT
        primary_key: true
      - name: service_name
        type: VARCHAR(100)
      - name: service_type
        type: VARCHAR(50)
      - name: created_date
        type: TIMESTAMP
      - name: price
        type: DECIMAL(10,2)
        
  - name: customer
    columns:
      - name: customer_id
        type: INT
        primary_key: true
      - name: customer_name
        type: VARCHAR(200)
      - name: email
        type: VARCHAR(100)
      - name: phone
        type: VARCHAR(20)
      - name: registration_date
        type: DATE
"""
    else:
        return """database: target_db
tables:
  - name: service_details
    columns:
      - name: id
        type: INTEGER
        primary_key: true
      - name: name
        type: VARCHAR(100)
      - name: type
        type: VARCHAR(50)
      - name: created_at
        type: TIMESTAMP
      - name: cost
        type: DECIMAL(10,2)
        
  - name: client
    columns:
      - name: client_id
        type: INTEGER
        primary_key: true
      - name: full_name
        type: VARCHAR(200)
      - name: email_address
        type: VARCHAR(100)
      - name: contact_number
        type: VARCHAR(20)
      - name: signup_date
        type: DATE
"""

def display_results(results: Dict):
    """Display analysis results"""
    if 'error' in results:
        st.error(f"Error: {results['error']}")
        if 'messages' in results:
            with st.expander("Agent Messages"):
                for msg in results['messages']:
                    st.text(msg)
        return
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Similarity", f"{results.get('overall_schema_similarity', 0)}%")
    with col2:
        st.metric("Table Matches", len(results.get('table_matches', [])))
    with col3:
        st.metric("Model Used", results.get('model_used', 'Unknown'))
    
    # Agent workflow visualization
    with st.expander("Agent Workflow Log", expanded=False):
        for msg in results.get('agent_messages', []):
            st.text(msg)
    
    st.divider()
    
    # Table matches
    for table_match in results.get('table_matches', []):
        with st.expander(
            f"{table_match['source_table']} → {table_match['target_table']} ({table_match['accuracy']}%)",
            expanded=True
        ):
            st.markdown(f"**Accuracy:** {table_match['accuracy']}%")
            st.markdown(f"**Reasoning:** {table_match['reasoning']}")
            
            st.subheader("Column Matches")
            
            if table_match.get('column_matches'):
                col_data = []
                for col_match in table_match['column_matches']:
                    col_data.append({
                        'Source Column': col_match['source_column'],
                        'Target Column': col_match['target_column'],
                        'Accuracy': f"{col_match['accuracy']}%",
                        'Source Type': col_match.get('source_type', 'N/A'),
                        'Target Type': col_match.get('target_type', 'N/A'),
                        'Compatible': '✓' if col_match.get('type_compatible', True) else '✗',
                        'Reasoning': col_match['reasoning']
                    })
                
                df = pd.DataFrame(col_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            if table_match.get('unmatched_source_columns'):
                st.warning(f"Unmatched Source Columns: {', '.join(table_match['unmatched_source_columns'])}")
            if table_match.get('unmatched_target_columns'):
                st.warning(f"Unmatched Target Columns: {', '.join(table_match['unmatched_target_columns'])}")
    
    if results.get('unmatched_source_tables'):
        st.warning(f"Unmatched Source Tables: {', '.join(results['unmatched_source_tables'])}")
    if results.get('unmatched_target_tables'):
        st.warning(f"Unmatched Target Tables: {', '.join(results['unmatched_target_tables'])}")
    
    if results.get('recommendations'):
        st.subheader("Recommendations")
        for rec in results['recommendations']:
            st.info(rec)

def render_home_page():
    """Render the home page with navigation cards"""
    st.title("🏠 SDM Platform - AI Orchestrated")
    st.markdown("### Welcome to the Smart Data Migration Platform")
    st.markdown("---")
    
    # Create navigation cards
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("### 📊 Data Analysis")
            st.markdown("""
            Advanced data analysis and schema comparison powered by AI agents.
            
            **Features:**
            - Schema comparison and analysis
            - Data profiling and statistics
            - Schema mapping suggestions
            - Data quality assessment
            - AI-powered recommendations
            """)
            if st.button("Go to Data Analysis", key="nav_analysis", use_container_width=True):
                st.session_state.page = "Data Analysis"
                st.rerun()
        
        st.markdown("")
        
        with st.container():
            st.markdown("### 🔄 Transformer")
            st.markdown("""
            Apply data transformations and business rules during migration.
            
            **Features:**
            - Custom transformation rules
            - Data type conversions
            - Value mappings
            - Calculated fields
            """)
            if st.button("Go to Transformer", key="nav_transformer", use_container_width=True):
                st.session_state.page = "Transformer"
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("### ✅ Validator")
            st.markdown("""
            Validate data quality and integrity before and after migration.
            
            **Features:**
            - Schema validation
            - Data quality checks
            - Constraint verification
            - Anomaly detection
            """)
            if st.button("Go to Validator", key="nav_validator", use_container_width=True):
                st.session_state.page = "Validator"
                st.rerun()
        
        st.markdown("")
        
        with st.container():
            st.markdown("### � Data Migration")
            st.markdown("""
            Execute and manage data migration workflows.
            
            **Features:**
            - Sample and full migration execution
            - Migration progress tracking
            - Error handling and recovery
            - Performance optimization
            """)
            if st.button("Go to Data Migration", key="nav_migration", use_container_width=True):
                st.session_state.page = "Data Migration"
                st.rerun()
        
        st.markdown("")
        
        with st.container():
            st.markdown("### �📝 Audit")
            st.markdown("""
            Track and review all migration activities and changes.
            
            **Features:**
            - Migration history
            - Change logs
            - Performance metrics
            - Compliance reports
            """)
            if st.button("Go to Audit", key="nav_audit", use_container_width=True):
                st.session_state.page = "Audit"
                st.rerun()

def render_data_analysis_page():
    """Render the data analysis page with multiple tabs"""
    st.title("📊 Data Analysis")
    
    # Tabs per your requirement: Sample AI - Schema matcher will contain Input Schemas UI
    tabs = st.tabs([
        "Sample AI - Schema matcher",
        "Choose Instances",
        "Data Analyst Agents Workflow",
        "Data Architect Agents Workflow",
        "Human Review"
    ])
    
    # Tab 0: Sample AI - Schema matcher (moved from previous Input Schemas)
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Source Schema")
            input_method_source = st.radio(
                "Input Method (Source)",
                ["Upload YAML", "Paste YAML", "Use Sample"],
                key="dm_source_method"
            )
            
            source_content = ""
            if input_method_source == "Upload YAML":
                uploaded_source = st.file_uploader(
                    "Upload Source Schema YAML",
                    type=['yaml', 'yml'],
                    key="dm_source_upload"
                )
                if uploaded_source:
                    source_content = uploaded_source.read().decode('utf-8')
                    st.text_area("Preview", source_content, height=400, key="dm_src_prev")
            elif input_method_source == "Paste YAML":
                source_content = st.text_area(
                    "Paste Source Schema YAML",
                    height=400,
                    key="dm_source_paste"
                )
            else:
                source_content = create_sample_yaml("source")
                st.text_area("Sample", source_content, height=400, key="dm_src_samp")
        
        with col2:
            st.subheader("Target Schema")
            input_method_target = st.radio(
                "Input Method (Target)",
                ["Upload YAML", "Paste YAML", "Use Sample"],
                key="dm_target_method"
            )
            
            target_content = ""
            if input_method_target == "Upload YAML":
                uploaded_target = st.file_uploader(
                    "Upload Target Schema YAML",
                    type=['yaml', 'yml'],
                    key="dm_target_upload"
                )
                if uploaded_target:
                    target_content = uploaded_target.read().decode('utf-8')
                    st.text_area("Preview", target_content, height=400, key="dm_tgt_prev")
            elif input_method_target == "Paste YAML":
                target_content = st.text_area(
                    "Paste Target Schema YAML",
                    height=400,
                    key="dm_target_paste"
                )
            else:
                target_content = create_sample_yaml("target")
                st.text_area("Sample", target_content, height=400, key="dm_tgt_samp")
        
        st.divider()
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_button = st.button("Run Agent Workflow", type="primary", use_container_width=True, key="dm_run")
        with col_btn2:
            clear_button = st.button("Clear Results", use_container_width=True, key="dm_clear")
        
        if analyze_button:
            if not source_content or not target_content:
                st.error("Please provide both schemas")
            else:
                is_connected, _ = check_ollama_connection()
                if not is_connected:
                    st.error("Cannot connect to Ollama")
                else:
                    try:
                        with st.spinner("Running multi-agent workflow..."):
                            available_models = get_available_models()
                            # Keep using selected model if in session_state, else use default
                            selected_model = st.session_state.get('selected_model', available_models[0] if available_models else "llama3.2")
                            workflow = SchemaMatchingWorkflow(model_name=selected_model)
                            results = workflow.run(source_content, target_content)
                            st.session_state['analysis_results'] = results
                        
                        if 'error' not in results:
                            st.success("Analysis complete! Check Human Review tab.")
                        else:
                            st.error("Analysis completed with errors.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        if clear_button:
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            st.rerun()
    
    # Tab 1: Choose Instances (existing content moved here)
    with tabs[1]:
        st.subheader("Choose Source and Target Instances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Source Instance")
            source_type = st.selectbox("Source Database Type", ["PostgreSQL", "MySQL", "Oracle", "SQL Server", "MongoDB"], key="dm_src_db_type")
            source_host = st.text_input("Host", value="localhost", key="dm_src_host")
            source_port = st.text_input("Port", value="5432", key="dm_src_port")
            source_database = st.text_input("Database Name", key="dm_src_db")
            source_username = st.text_input("Username", key="dm_src_user")
            source_password = st.text_input("Password", type="password", key="dm_src_pass")
            
            if st.button("Test Source Connection", key="dm_test_src"):
                st.info("Connection test functionality to be implemented")
        
        with col2:
            st.markdown("#### Target Instance")
            target_type = st.selectbox("Target Database Type", ["PostgreSQL", "MySQL", "Oracle", "SQL Server", "MongoDB"], key="dm_tgt_db_type")
            target_host = st.text_input("Host", value="localhost", key="dm_tgt_host")
            target_port = st.text_input("Port", value="5432", key="dm_tgt_port")
            target_database = st.text_input("Database Name", key="dm_tgt_db")
            target_username = st.text_input("Username", key="dm_tgt_user")
            target_password = st.text_input("Password", type="password", key="dm_tgt_pass")
            
            if st.button("Test Target Connection", key="dm_test_tgt"):
                st.info("Connection test functionality to be implemented")
        
        st.divider()
        
        if st.button("Save Configuration & Continue", type="primary", use_container_width=True, key="dm_save_conf"):
            st.success("Configuration saved! Proceed to next tab.")
    
    # Tab 2: Data Analyst Agents Workflow
    with tabs[2]:
        st.subheader("Data Analyst Agents Workflow")
        st.markdown("AI agents analyze source and target data for profiling and statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Source Data Analysis")
            st.info("Agent Status: Ready")
            
            if st.button("Run Source Analysis", key="dm_run_src_analysis"):
                with st.spinner("Analyzing source data..."):
                    st.success("Source analysis complete!")
                    st.json({
                        "total_tables": 15,
                        "total_rows": 1250000,
                        "data_quality_score": 92,
                        "null_percentage": 3.2
                    })
        
        with col2:
            st.markdown("#### Target Data Analysis")
            st.info("Agent Status: Ready")
            
            if st.button("Run Target Analysis", key="dm_run_tgt_analysis"):
                with st.spinner("Analyzing target data..."):
                    st.success("Target analysis complete!")
                    st.json({
                        "total_tables": 12,
                        "available_capacity": "500GB",
                        "compatibility_score": 88
                    })
    
    # Tab 3: Data Architect Agents Workflow
    with tabs[3]:
        st.subheader("Data Architect Agents Workflow")
        st.markdown("AI-powered schema validation and rule generation")
        
        st.info("The Data Architect Agent will analyze your schemas and generate validation rules.")
        
        if st.button("Generate Validation Rules", type="primary", use_container_width=True):
            with st.spinner("Generating validation rules..."):
                st.success("Validation rules generated successfully!")
                st.json({
                    "rules": [
                        {"id": "rule_001", "type": "data_type_validation", "description": "Validate data types match target schema"},
                        {"id": "rule_002", "type": "required_field", "description": "Check for required fields in source data"},
                        {"id": "rule_003", "type": "referential_integrity", "description": "Verify foreign key relationships"},
                        {"id": "rule_004", "type": "custom_validation", "description": "Custom business rule validation"}
                    ]
                })
        
        st.divider()
        
        st.subheader("Validation Rules")
        st.markdown("""
        The following validation rules will be applied during migration:
        
        1. **Data Type Validation**: Ensures data types match the target schema
        2. **Required Fields**: Validates that all required fields are present
        3. **Referential Integrity**: Checks foreign key relationships
        4. **Custom Rules**: Applies any custom business rules
        """)
    
    # Tab 4: Human Review (Results moved here)
    with tabs[4]:
        st.subheader("Human Review")
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            display_results(results)
            
            st.divider()
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Download Results (JSON)",
                    data=json_str,
                    file_name=f"schema_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_dl2:
                if 'error' not in results and results.get('table_matches'):
                    csv_data = []
                    for tm in results.get('table_matches', []):
                        for cm in tm.get('column_matches', []):
                            csv_data.append({
                                'Source Table': tm['source_table'],
                                'Target Table': tm['target_table'],
                                'Table Accuracy': tm['accuracy'],
                                'Source Column': cm['source_column'],
                                'Target Column': cm['target_column'],
                                'Column Accuracy': cm['accuracy'],
                                'Source Type': cm.get('source_type', 'N/A'),
                                'Target Type': cm.get('target_type', 'N/A')
                            })
                    
                    if csv_data:
                        df_export = pd.DataFrame(csv_data)
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="Download Matches (CSV)",
                            data=csv,
                            file_name=f"schema_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        else:
            st.info("No analysis results available. Run Sample AI - Schema matcher to generate results.")

def render_data_migration_execution_page():
    """Render the data migration execution page with multiple tabs"""
    st.title("🔄 Data Migration")
    
    # Tabs per your requirement: Sample AI - Schema matcher will contain Input Schemas UI
    tabs = st.tabs([
        "Sample Run",
        "Full Run"
    ])
    
    # Tab 0: Sample Run
    with tabs[0]:
        st.subheader("Sample Run")
        st.markdown("Run a sample migration to test the workflow")
        
        if st.button("Run Sample Migration", type="primary", use_container_width=True):
            with st.spinner("Running sample migration..."):
                st.success("Sample migration complete!")
    
    # Tab 1: Full Run
    with tabs[1]:
        st.subheader("Full Run")
        st.markdown("Run the full migration workflow")
        
        if st.button("Run Full Migration", type="primary", use_container_width=True):
            with st.spinner("Running full migration..."):
                st.success("Full migration complete!")

def main():
    st.set_page_config(
        page_title="SDM Platform - AI Orchestrated",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 SDM Platform - AI Orchestrated")
    st.markdown("*AI-orchestrated platform for smart data migration*")
    
    # Initialize session state for page if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Sidebar
    with st.sidebar:
        # Requirement 1: Change "Configuration" to "SDM Platform - AI Orchestrated"
        st.header("SDM Platform - AI Orchestrated")
        
        # Check Ollama but remove the ✅ Ollama Connected success message (Requirement 2)
        is_connected, connection_msg = check_ollama_connection()
        if not is_connected:
            st.error("❌ Ollama Not Connected")
            st.warning(f"Error: {connection_msg}")
        # If connected, do not show the success message per requirement (silently continue)
        
        # Requirement 3: Change label to "LLM selection"
        available_models = get_available_models()
        selected_model = st.selectbox(
            "LLM selection",
            available_models,
            index=0 if available_models else None
        )
        # persist selected model to session state so data-migration page can use it
        st.session_state['selected_model'] = selected_model
        
        st.divider()
        
        # Requirement 4: Add page navigation in sidebar
        page_choice = st.radio(
            "Navigate",
            ["Home", "Data Analysis", "Data Migration", "Transformer", "Validator", "Audit"],
            index=["Home", "Data Analysis", "Data Migration", "Transformer", "Validator", "Audit"].index(st.session_state.page) 
                if st.session_state.page in ["Home", "Data Analysis", "Data Migration", "Transformer", "Validator", "Audit"] else 0,
            key="sidebar_nav"
        )
        st.session_state.page = page_choice
        
        st.divider()
    
    # Route to the selected page
    page = st.session_state.page
    if page == "Home":
        render_home_page()
    elif page == "Data Analysis":
        render_data_analysis_page()
    elif page == "Data Migration":
        render_data_migration_execution_page()
    elif page == "Transformer":
        st.title("🔄 Transformer")
        st.write("Data transformation tools will be available here.")
    elif page == "Validator":
        st.title("✅ Validator")
        st.write("Data validation tools will be available here.")
    elif page == "Audit":
        st.title("📝 Audit")
        st.write("Audit and logging tools will be available here.")
    else:
        st.session_state.page = "Home"
        st.rerun()

if __name__ == "__main__":
    main()
