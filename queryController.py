import pandas as pd
from DBController import get_connection
from datetime import timedelta
import re
import os
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import numpy as np
import streamlit as st

os.environ["AZURE_API_KEY"] = st.secrets["AZURE_API_KEY"]
os.environ["AZURE_API_BASE"] = st.secrets["AZURE_API_BASE"]

# Initialize the Azure Chat LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",           # Your Azure deployment name
    model_name="azure/gpt-4.1-nano",           # Usually same as deployment
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2025-01-01-preview",          # API version
    temperature=0,                              # deterministic rating
    streaming=True,                            # No streaming for evaluation
    model_kwargs={"stream_options": {"include_usage": True}}
)

def evaluate_trace_relevance(input_text, output_text, span_id):
    """
    Runs LLM evaluation for a given trace using AzureChatOpenAI,
    extracts a 1–10 relevance score, and returns it.
    """
    prompt = f"""
You are an expert evaluator. Analyze the following model interaction:

Input:
{input_text}

Output:
{output_text}

Rate how relevant the output is to the input on a scale of 1 (irrelevant) to 10 (highly relevant).
Respond ONLY with a number between 1 and 10.
"""

    try:
        # Call the Azure Chat LLM
        response = llm(
            messages=[
                SystemMessage(content="You are a precise evaluator that responds with only a numeric rating."),
                HumanMessage(content=prompt)
            ]
        )

        model_reply = response.content.strip()

        # Extract numeric score
        match = re.search(r"\b([1-9]|10)\b", model_reply)
        if not match:
            raise ValueError(f"Could not parse numeric score from model output: {model_reply}")

        score = int(match.group(1))
        return score

    except Exception as e:
        print(f"[ERROR] Failed to evaluate span_id {span_id}: {e}")
        return None


class LLMQueries:

    def get_dropdown_options(self):
        query = "SELECT DISTINCT name as category FROM projects ORDER BY name"
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
        return df['category'].tolist()

    def get_date_range_for_project(self, project_name):
        query = """
        SELECT MIN(start_time) AS min_date,
               MAX(start_time) AS max_date
        FROM spans
        WHERE trace_rowid IN (
            SELECT id FROM traces
            WHERE project_rowid IN (
                SELECT id FROM projects WHERE name = %s
            )
        )
        """
        with get_connection() as conn:
            df = pd.read_sql(query, conn, params=(project_name,))
            return df['min_date'][0].date(), df['max_date'][0].date()
            

    def get_tile1_value(self, selected, start_date, end_date):
        query = """
        SELECT COUNT(*) AS value
        FROM spans
        WHERE span_kind = 'LLM'
          AND start_time >= %s AND 
          start_time< %s
          AND trace_rowid IN (
              SELECT id FROM traces WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
           )   
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date+timedelta(days=1), selected))['value'][0]

    def get_tile2_value(self, selected, start_date, end_date):
        query = """
        SELECT AVG(EXTRACT(EPOCH FROM end_time - start_time)) AS value 
        FROM spans 
        WHERE start_time >= %s AND 
          start_time< %s
          AND trace_rowid IN (
              SELECT id FROM traces 
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date+timedelta(days=1), selected))['value'][0]

    def get_tile3_value(self, selected, start_date, end_date):
        query = """
        SELECT SUM(llm_token_count_prompt) AS value 
        FROM spans 
        WHERE start_time >= %s AND 
          start_time< %s
          AND trace_rowid IN (
              SELECT id FROM traces 
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date+timedelta(days=1), selected))['value'][0]

    def get_tile4_value(self, selected, start_date, end_date):
        query = """
        SELECT SUM(llm_token_count_completion) AS value 
        FROM spans 
        WHERE start_time >= %s AND 
          start_time< %s
          AND trace_rowid IN (
              SELECT id FROM traces 
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date+timedelta(days=1), selected))['value'][0]

    def get_metric1_data(self, selected, start_date, end_date):
        query = """
        SELECT DATE_TRUNC('hour', start_time) as hour,
               COUNT(DISTINCT trace_rowid) AS count
        FROM spans
        WHERE span_kind = 'LLM'
          AND start_time >= %s AND 
          start_time< %s
          AND trace_rowid IN (
              SELECT id FROM traces
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        GROUP BY hour
        ORDER BY hour;
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date+timedelta(days=1), selected))

    def get_metric2_data(self, selected, start_date, end_date):
        query = """
        SELECT DATE(start_time) AS date,
       status_code AS status,
       COUNT(*) AS trace_count
        FROM spans
        WHERE start_time >= %s AND 
          start_time< %s
        AND trace_rowid IN (
              SELECT id FROM traces
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
         )     
        GROUP BY date, status
        ORDER BY date, status;
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date+timedelta(days=1), selected))
   
    def getAttributes(self,selected,start_date, end_date):
     query = """
        SELECT p.name,s.trace_rowid,s.name as span_name, s.start_time,s.end_time,s.attributes, s. span_kind, 
        s.span_id,s.parent_id,t.start_time as timestamp,
        s.cumulative_error_count,s.cumulative_llm_token_count_prompt,s.cumulative_llm_token_count_completion,s.status_code, s.status_message
        FROM projects p
        JOIN traces t ON p.id = t.project_rowid
        JOIN spans s ON t.id = s.trace_rowid
        WHERE p.name = %s
        AND s.start_time >= %s AND 
          s.start_time< %s
        """
     with get_connection() as conn:
        return pd.read_sql(query, conn, params=( selected,start_date, end_date+timedelta(days=1)))

    def get_trace_ids(self, selected, start_date, end_date):
        """
        Fetch all trace IDs for a given project and date range.
        """
        query = """
        select s.id AS id, s.parent_id as parent_id, s.attributes as attributes, parent_span.attributes->'graph'->'node'->>'id' as agent_name, EXTRACT(EPOCH FROM (s.end_time - parent_span.start_time)) AS latency, s.status_code from spans as s 
        left join traces as t 
        on s.trace_rowid = t.id 
        left join projects as p on t.project_rowid = p.id 
        LEFT JOIN spans AS parent_span
        ON s.parent_id = parent_span.span_id
        where 
        p.name = %s
        and s.id not in (select aes.span_serial_id from agent_eval_score aes)
        AND s.start_time >= %s
        AND s.start_time < %s
        AND s.id = (
        SELECT MAX(s2.id)
        FROM spans AS s2
        WHERE s2.parent_id = s.parent_id
        and s2.name = 'completion'
        )
        AND parent_span.span_kind ='AGENT'
        and parent_span.attributes->'graph'->'node'->>'id' is not null

        """
        with get_connection() as conn:
            df = pd.read_sql(
                query,
                conn,
                params=(selected, start_date, end_date + timedelta(days=1))
            )
        # df["agent_name"] = df["agent_name"].str.split(".").str[0]
        df["input_data"] = df["attributes"].apply(
            lambda x: x.get("llm", {}).get("input_messages", [{}])[0].get("message", {})
        )

        # # Extract output_data
        df["output_data"] = df["attributes"].apply(
            lambda x: x.get("llm", {}).get("output_messages", [{}])[0].get("message", {})
        )
        df["total_token_count"] = df["attributes"].apply(
            lambda x: x.get("llm", {}).get("token_count", {}).get("total", {})
        )

        return df

    def evaluate_all_traces(self, selected, start_date, end_date):
        """
        Get trace IDs and run evaluation for each.
        """
        df = self.get_trace_ids(selected, start_date, end_date)
        if df.empty:
            print("No traces found for given filters.")
            # return pd.DataFrame()

        results = []
        for index, row in df.iterrows():
            try:
                # Example: Call your external evaluation function here

                # Fetch input/output for this trace (or however you get them)

                score = evaluate_trace_relevance(
                    input_text=row['input_data'],
                    output_text=row['output_data'],
                    span_id=row['parent_id']
                )
                token_count = row["total_token_count"]
                print(type(token_count))
                if token_count == {}:
                    print("inside token count condition")
                    token_count = 0
                input_data = row['input_data']
                output_data = row['output_data']
                # Convert to JSON only if they are dicts/lists
                if isinstance(input_data, (dict, list)):
                    print("inside json type")
                    input_data = json.dumps(input_data)

                if isinstance(output_data, (dict, list)):
                    print("inside json type")
                    output_data = json.dumps(output_data)
                print(score)
                results.append((
                    row['id'],
                    json.dumps({"gpt-4o-mini": score}),
                    json.dumps(row['input_data']),
                    json.dumps(row['output_data']),
                    row['latency'],
                    token_count,
                    row["agent_name"]
                ))
                # break
            except Exception as e:
                print(f"⚠️ Error evaluating trace {row["parent_id"]}: {e}")
        print(results)
        print(len(results))
        query = """
        INSERT INTO agent_eval_score (span_serial_id,agent_score,input_data,output_data,latency,total_token_count,agent_name)
    	VALUES (%s,%s::jsonb,%s,%s,%s,%s,%s);"""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, results)
            conn.commit()  # Commit **once** after all rows
        # try:
        #     with get_connection() as conn:
        #         with conn.cursor() as cur:
        #             for row in results:
        #                 cur.execute(query, row)
        #                 conn.commit()
        # except Exception as e:
        #     print("[DB ERROR]", e)
        # return results
    def get_agent_score_all_data(self, selected, start_date, end_date):
        self.evaluate_all_traces(selected, start_date, end_date)
        query = """
        SELECT aes.*, s.status_code, s.start_time FROM agent_eval_score aes
        left join spans as s 
        on aes.span_serial_id = s.id
        left join traces as t 
        on s.trace_rowid = t.id 
        left join projects as p on t.project_rowid = p.id 
            WHERE p.name = %s
            AND s.start_time >= %s AND 
            s.start_time< %s
            """
        with get_connection() as conn:
            df = pd.read_sql(query, conn, params=( selected,start_date, end_date+timedelta(days=1)))

            df["llm_eval_scores"] = df["agent_score"].apply(self.extract_llm_score)

            # --- Step 2: Ensure numeric types ---
            df["latency_ms"] = pd.to_numeric(df["latency"], errors="coerce")
            df["token_count"] = pd.to_numeric(df["total_token_count"], errors="coerce")

            # --- Step 3: Create error flag ---
            # Assuming non-200 status codes = errors
            df["error"] = df["status_code"].apply(lambda x: 0 if x == 'OK' else 1)
        return df

    def sumarize_agent_score(self, df):
        # --- Step 4: Group by agent ---
        grouped = df.groupby("agent_name").agg(
            avg_llm_score=("llm_eval_scores", "mean"),
            avg_latency=("latency_ms", "mean"),
            token_std_dev=("token_count", "std"),
            error_rate=("error", "mean")
        ).reset_index()

        # --- Step 5: Compute composite score ---
        grouped["composite_score"] = (
            0.5 * grouped["avg_llm_score"]
            + 0.2 * (-grouped["avg_latency"])
            + 0.2 * (-grouped["token_std_dev"])
            + 0.1 * (1 - grouped["error_rate"])
        )

        # --- Step 6: Normalize composite scores to [1,10] scale ---
        min_score = grouped["composite_score"].min()
        max_score = grouped["composite_score"].max()

        if max_score == min_score:
            grouped["normalized_score"] = 10.0
        else:
            grouped["normalized_score"] = 1 + 9 * (
                (grouped["composite_score"] - min_score)
                / (max_score - min_score)
            )

        # --- Step 7: Sort descending by normalized score (optional) ---
        grouped = grouped.sort_values("normalized_score", ascending=False)
        return grouped

    def extract_llm_score(self, agent_score):
        if pd.isna(agent_score):
            return np.nan
        try:
            if isinstance(agent_score, str):
                data = json.loads(agent_score)
            elif isinstance(agent_score, dict):
                data = agent_score
            else:
                return np.nan
            # Take first numeric value (e.g., {"gpt-4o-mini": 10})
            return next(iter(data.values()))
        except Exception:
            return np.nan
