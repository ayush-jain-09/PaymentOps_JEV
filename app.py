import streamlit as st
import src.config as config

st.set_page_config(
    page_title="PaymentOps Agent",
    page_icon="\U0001f4b3",
    layout="wide",
)


@st.cache_resource
def load_laya():
    """Load Laya router once at startup."""
    import src.laya_router as lr
    return lr.LAYA_AVAILABLE


@st.cache_resource
def load_graph():
    from src.graph import get_graph
    return get_graph()


def badge(label: str, color: str) -> str:
    colors = {
        "green": "#28a745",
        "orange": "#fd7e14",
        "red": "#dc3545",
        "blue": "#007bff",
        "gray": "#6c757d",
    }
    bg = colors.get(color, colors["gray"])
    return f'<span style="background:{bg};color:white;padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:600">{label}</span>'


def decision_color(decision: str) -> str:
    mapping = {
        "safe": "green",
        "completed": "green",
        "refunded": "green",
        "requires_approval": "orange",
        "low_confidence": "orange",
        "blocked": "red",
        "failed": "red",
        "none": "gray",
        "pending": "blue",
    }
    return mapping.get(decision.lower(), "gray")


def render_trace(state: dict):
    """Render the decision trace panel."""
    trace = state.get("trace", [])

    for entry in trace:
        step = entry.get("step", "")

        if step == "USER":
            st.markdown("**\U0001f464 USER**")
            st.info(entry.get("query", ""))

        elif "TOOL ROUTER" in step:
            st.markdown(f"**\U0001f9e0 {step}**")
            decision = entry.get("decision", "")
            conf = entry.get("confidence", 0.0)
            color = decision_color(decision)
            st.markdown(badge(decision, color), unsafe_allow_html=True)
            st.progress(min(conf, 1.0), text=f"Confidence: {conf:.0%}")
            probs = entry.get("probabilities", {})
            if probs:
                with st.expander("Probabilities"):
                    import pandas as pd
                    df = pd.DataFrame(list(probs.items()), columns=["Label", "Probability"])
                    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
                    st.dataframe(df, use_container_width=True)
            if entry.get("error"):
                st.caption(f"\u26a0\ufe0f {entry['error']}")

        elif "ACTION GATE" in step:
            st.markdown(f"**\U0001f6aa {step}**")
            decision = entry.get("decision", "")
            conf = entry.get("confidence", 0.0)
            color = decision_color(decision)
            st.markdown(badge(decision, color), unsafe_allow_html=True)
            st.progress(min(conf, 1.0), text=f"Confidence: {conf:.0%}")
            probs = entry.get("probabilities", {})
            if probs:
                with st.expander("Gate Probabilities"):
                    import pandas as pd
                    df = pd.DataFrame(list(probs.items()), columns=["Label", "Probability"])
                    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
                    st.dataframe(df, use_container_width=True)

        elif step == "SYSTEM":
            msg = entry.get("message", "")
            if "NOT executed" in msg or "blocked" in msg.lower():
                st.error(f"\U0001f6ab SYSTEM: {msg}")
            elif "approval" in msg.lower() or "below threshold" in msg.lower():
                st.warning(f"\u26a0\ufe0f SYSTEM: {msg}")
            else:
                st.info(f"\u2139\ufe0f SYSTEM: {msg}")

        elif step == "TOOL EXECUTION":
            tool = entry.get("tool", "")
            result = entry.get("result", {})
            st.markdown(f"**\U0001f527 TOOL: `{tool}`**")
            with st.expander("Tool Result"):
                st.json(result)

        elif step == "LLM":
            st.markdown("**\U0001f916 LLM RESPONSE**")
            st.success(entry.get("response", ""))

        st.markdown("---")


def main():
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar
    with st.sidebar:
        st.title("\U0001f4b3 PaymentOps Agent")
        st.caption("Decision Model + LLM Architecture")
        st.divider()

        st.subheader("Settings")
        threshold = st.slider(
            "Routing Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=config.ROUTING_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Queries with Laya confidence below this value fall back to clarification.",
        )
        config.ROUTING_CONFIDENCE_THRESHOLD = threshold

        st.divider()
        st.subheader("Sample Queries")
        samples = [
            "What amount was charged for TX-1042?",
            "Why did TX-1003 fail?",
            "Calculate 80% of \u20b94999",
            "Refund TX-1042",
            "What is a settlement mismatch?",
        ]
        for sample in samples:
            if st.button(sample, use_container_width=True):
                st.session_state["pending_query"] = sample

        st.divider()
        laya_ok = load_laya()
        if laya_ok:
            st.success("\u2705 Laya Router loaded")
        else:
            st.error("\u274c Laya Router unavailable")

    # Main layout
    col_chat, col_trace = st.columns([6, 4])

    with col_chat:
        st.header("Chat")

        # Display conversation history
        for q, s in st.session_state.history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(s.get("llm_response", "(no response)"))

        # Input form
        with st.form(key="query_form", clear_on_submit=True):
            default_val = st.session_state.pop("pending_query", "")
            user_query = st.text_input(
                "Enter your query:",
                value=default_val,
                placeholder="e.g. What happened with TX-1042?",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_query.strip():
            with st.spinner("Processing..."):
                try:
                    from src.graph import run_agent
                    result_state = run_agent(user_query.strip())
                    st.session_state.history.append((user_query.strip(), result_state))
                    st.session_state["last_state"] = result_state
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent error: {e}")

    with col_trace:
        st.header("Decision Trace")
        if "last_state" in st.session_state:
            render_trace(st.session_state["last_state"])
        elif not st.session_state.history:
            st.info("Submit a query to see the decision trace here.")
        else:
            render_trace(st.session_state.history[-1][1])


if __name__ == "__main__":
    main()
