import streamlit as st
import json
import requests

st.set_page_config(page_title="MCP Server Tester", layout="wide")

st.title("MCP Server Tester")

# ==================================================
# SIDEBAR: MCP SERVER CONFIGURATION
# ==================================================
st.sidebar.header("MCP Server Configuration")

# MCP Server URL
base_url = st.sidebar.text_input("Enter MCP Server URL:","")

# Choose header input method
header_mode = st.sidebar.radio(
    "Choose Header Mode:",
    ["Key-Value Pairs", "Custom JSON"],
    key="header_mode"
)


# Initialize header store
if "headers_list" not in st.session_state:
    st.session_state.headers_list = [{"key": "Content-Type", "value": "application/json"},
    {"key": "Accept", "value": "application/json, text/event-stream"},
    {"key": "x-api-key", "value": ""}]

# --- Key-Value Mode ---
if header_mode == "Key-Value Pairs":
    st.sidebar.subheader("HTTP Headers")

    # Display dynamic header fields
    for i, header in enumerate(st.session_state.headers_list):
        cols = st.sidebar.columns([2, 3, 1])
        
        with cols[0]:
            header["key"] = st.text_input(f"Key {i+1}", header["key"], key=f"key_{i}")
        
        with cols[1]:
            header["value"] = st.text_input(f"Value {i+1}", header["value"], key=f"value_{i}")
        
        with cols[2]:
            # Add a small spacer to roughly center the button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("X", key=f"remove_{i}"):
                st.session_state.headers_list.pop(i)
                st.rerun()

    # Add new header button
    if st.sidebar.button("Add Header"):
        st.session_state.headers_list.append({"key": "", "value": ""})
        st.rerun()

    # Convert to dict
    headers = {h["key"]: h["value"] for h in st.session_state.headers_list if h["key"]}

# --- Custom JSON Mode ---
else:
    st.sidebar.subheader("Custom Header JSON")
    custom_headers_json = st.sidebar.text_area(
        "Enter Headers as JSON:",
        value=json.dumps(
            {"Accept": "application/json, text/event-stream"}, indent=2
        ),
        height=150,
        key="custom_headers_json",
    )
    try:
        headers = json.loads(custom_headers_json)
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON format.")
        headers = {"Accept": "application/json, text/event-stream"}

# Show constructed headers
with st.sidebar.expander("🔍 View Constructed Headers"):
    st.json(headers)



# -----------------------------
# Initialize & Create Session
# -----------------------------
st.sidebar.subheader("Initialize MCP Session")

init_payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
        "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
}

if st.sidebar.button("Initialize and Create Session", key="init_btn"):
    try:
        # Step 1: Initialize
        init_resp = requests.post(base_url, headers=headers, json=init_payload)
        # st.sidebar.write("🔄 Initialization Response:", init_resp.text)

        # Extract 'data:' line
        raw_text = init_resp.text
        data_line = next((l for l in raw_text.splitlines() if l.startswith("data:")), None)

        if not data_line:
            st.sidebar.error("Initialization failed — no 'data:' line found in response.")
        else:
            init_data = json.loads(data_line[len("data: "):])

            # Step 2: Create Session
            session_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/create",
                "params": init_data["result"],
            }

            session_resp = requests.post(base_url, headers=headers, json=session_payload, timeout=30)
            session_id = session_resp.headers.get("mcp-session-id")

            if session_id:
                st.session_state.session_id = session_id
                # st.sidebar.success(f"✅ Session created successfully! ID: {session_id}")

                session_headers = {**headers, "Mcp-Session-Id": session_id}

                # Step 3: Re-initialize with Session ID
                reinit_resp = requests.post(base_url, headers=session_headers, json=init_payload, timeout=30)
                # st.sidebar.caption("🔁 Re-initialize response:")
                # st.sidebar.text(reinit_resp.text)
                st.sidebar.success(f"MCP Server successfully connected")
            else:
                st.sidebar.warning("No session ID found in headers.")
    except Exception as e:
        st.sidebar.error(f"Error initializing session: {e}")

# ==================================================
# STEP 2: List Tools (Optional)
# ==================================================
st.subheader("List and Test MCP Tools")

if "session_id" not in st.session_state or not st.session_state.session_id:
    st.warning("Please initialize and create a session first.")
else:
    if st.button("List MCP Tools", key="list_tools_btn"):
        try:
            active_headers = {**headers, "Mcp-Session-Id": st.session_state.session_id}
            list_resp = requests.post(
                base_url,
                headers=active_headers,
                json={"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}},
                timeout=60,
            )

            # st.write("Tools Response:", list_resp.text)
            raw_text = list_resp.text

            # --- Extract data from SSE 'data:' lines ---
            data_line = next((l for l in raw_text.splitlines() if l.startswith("data:")), None)
            if data_line:
                try:
                    parsed_json = json.loads(data_line[len("data: "):])
                    tools_data = parsed_json.get("result", {}).get("tools", [])
                except Exception as e:
                    st.error(f"Error parsing 'data:' JSON: {e}")
                    tools_data = []
            else:
                try:
                    tools_data = list_resp.json().get("result", {}).get("tools", [])
                except Exception:
                    st.error("Could not extract tools from response.")
                    tools_data = []

            st.session_state.tools_data = tools_data
            st.session_state.active_tool = None
            st.session_state.tool_result = None

            if tools_data:
                st.success(f"✅ {len(tools_data)} tools found.")

        except Exception as e:
            st.error(f"Error listing tools: {e}")

# --------------------------------------------------
# Display tools and test them inline (persistent result)
# --------------------------------------------------
if "tools_data" in st.session_state and st.session_state.tools_data:
    for tool in st.session_state.tools_data:
        with st.expander(f"🧩 {tool['name']}", expanded=(st.session_state.get("active_tool") == tool["name"])):
            st.write(tool["description"])

            input_schema = tool.get("inputSchema", {}).get("properties", {})
            st.code(json.dumps(input_schema, indent=2), language="json")

            # JSON input box for arguments
            tool_args = st.text_area(
                f"Enter arguments for {tool['name']}:",
                value="{}",
                key=f"args_{tool['name']}",
                height=150,
            )

            run_clicked = st.button(f"▶️ Run {tool['name']}", key=f"run_{tool['name']}")
            if run_clicked:
                try:
                    parsed_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for tool arguments.")
                    st.stop()

                payload = {
                    "jsonrpc": "2.0",
                    "id": 100,
                    "method": "tools/call",
                    "params": {"name": tool["name"], "arguments": parsed_args},
                }

                try:
                    test_resp = requests.post(
                        base_url,
                        headers={**headers, "Mcp-Session-Id": st.session_state.session_id},
                        json=payload,
                        timeout=60,
                    )
                    st.write("Raw Response:", test_resp.text)

                    # Parse SSE data or fallback
                    data_line = next(
                        (l for l in test_resp.text.splitlines() if l.startswith("data:")), None
                    )
                    if data_line:
                        try:
                            parsed_result = json.loads(data_line[len("data: "):])
                        except Exception:
                            parsed_result = {"raw": test_resp.text}
                    else:
                        try:
                            parsed_result = test_resp.json()
                        except Exception:
                            parsed_result = {"raw": test_resp.text}

                    # ✅ Store persistent result
                    st.session_state.active_tool = tool["name"]
                    st.session_state.tool_result = parsed_result
                    st.rerun()

                except Exception as e:
                    st.error(f"Error testing tool: {e}")

            # ✅ Display persistent results after rerun
            if st.session_state.get("active_tool") == tool["name"] and st.session_state.get("tool_result"):
                st.markdown("#### Tool Output")
                try:
                    st.json(st.session_state["tool_result"])
                except Exception:
                    st.text(st.session_state["tool_result"])
        # except Exception as e:
        #     st.error(f"Error parsing tools list: {e}")


# ==================================================
# STEP 3: Test Tool
# ==================================================
# st.subheader("🧪 Step 3: Test a Tool")

# if "session_id" not in st.session_state:
#     st.warning("⚠️ Please create a session first before testing tools.")
# else:
#     tool_name = st.text_input("Enter Tool Name:", key="tool_name_field")

#     input_schema_hint = ""
#     if "tools_data" in st.session_state and tool_name:
#         for tool in st.session_state.tools_data:
#             if tool["name"] == tool_name:
#                 input_schema_hint = json.dumps(
#                     tool.get("inputSchema", {}).get("properties", {}), indent=2
#                 )
#                 break

#     st.caption("Enter tool arguments as JSON:")
#     tool_args = st.text_area(
#         "Tool Arguments",
#         value="{}",
#         key="tool_args_field"
#     )

#     if input_schema_hint:
#         with st.expander("📘 Input Schema Suggestion"):
#             st.code(input_schema_hint, language="json")

#     if st.button("Run Tool", key="run_tool_btn"):
#         try:
#             parsed_args = json.loads(tool_args)
#         except json.JSONDecodeError:
#             st.error("❌ Invalid JSON format in tool arguments.")
#             st.stop()

#         payload = {
#             "jsonrpc": "2.0",
#             "id": 100,
#             "method": "tools/call",
#             "params": {"name": tool_name, "arguments": parsed_args},
#         }

#         st.write("📤 Sending Request:", payload)

#         try:
#             test_resp = requests.post(
#                 base_url,
#                 headers={**headers, "Mcp-Session-Id": st.session_state.session_id},
#                 json=payload,
#             )
#             st.write("📥 Response:", test_resp.text)

#             try:
#                 st.json(test_resp.json())
#             except Exception:
#                 st.text(test_resp.text)
#         except Exception as e:
#             st.error(f"❌ Error testing tool: {e}")
