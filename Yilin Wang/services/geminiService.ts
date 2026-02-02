import { GoogleGenAI } from "@google/genai";
import { AnalystContext } from "../types";

const SYSTEM_PROMPT = `
You are a buy-side Fundamental Analyst Agent. Your job is to produce an auditable FCFF-DCF valuation and a 1–2 page investment memo for MSFT with a focus on capital structure and WACC. You must use tools for all numeric computations. You must never fabricate numbers. If a metric is missing, state it is unavailable and add it to limitations. Your memo must only use numbers contained in the provided context object. Use conditional language and explicitly disclose approximations (e.g., book debt proxy, Kd assumption, FCFF proxy).
`;

const MEMO_TEMPLATE_INSTRUCTION = `
Output the final memo in Markdown exactly following the memo template headings:

### Title
**MSFT — Fundamental Analyst Memo (FCFF DCF & Capital Structure)**  
Date (UTC): {{meta.generated_at_utc}}  
Data source: yfinance (cache_used={{meta.cache_used}})

### 1) Executive Summary
- Recommendation (e.g., Buy/Hold/Sell) with conditional language.
- Intrinsic value (base) and upside/downside vs spot (if available).
- One-paragraph rationale referencing **only** context numbers.

### 2) Business & Financial Snapshot (Evidence-based)
- Brief business positioning (sector/industry) if available.
- FCF history trend (use fcf_history summary).
- Any key stability/volatility note if computed.

### 3) Capital Structure & WACC (MANDATORY)
**Capital Structure**
- Equity (market cap E): {{capital_structure.equity_market_value_E}}
- Debt (book D): {{capital_structure.debt_book_value_D}}  (state proxy assumption if applicable)
- Cash: {{capital_structure.cash_and_equivalents}}
- Net debt: {{capital_structure.net_debt}}
- Weights: wE={{wacc.weights.w_e}}, wD={{wacc.weights.w_d}}

**Cost of Equity (CAPM)**
- rf={{wacc.rf}}, ERP={{wacc.erp}}, beta={{wacc.beta_used}}
- Ke={{wacc.cost_of_equity_ke}}
Explain why rf is user-input; explain beta source/override if used.

**Cost of Debt & Tax Shield**
- Kd={{wacc.cost_of_debt_kd}} and effective tax rate T={{wacc.tax_rate_effective}}
- After-tax Kd = Kd(1-T) (do NOT compute if not provided by tools; only reference tool outputs)

**WACC**
- WACC (base)={{wacc.wacc}}
- Brief sensitivity intuition: valuation impact when WACC changes (reference sensitivity block).

### 4) Valuation (FCFF DCF)
- Method: {{fcf.method}}
- FCF0: {{fcf.fcf0_latest}}
- Assumptions: years={{inputs.forecast_years}}, growth={{inputs.fcf_growth}}, terminal g={{inputs.terminal_g}}, WACC={{wacc.wacc}}
- **SCENARIO TABLE**:
  Create a standard Markdown table with proper header and alignment rows (e.g. \`|---|---|...\`) for the Base/Bull/Bear scenarios.
  Columns: **Scenario**, **WACC**, **FCF Growth**, **Value Per Share**, **Upside/Downside**.
  Ensure there is an empty line before and after the table.
- WACC×g sensitivity: describe key ranges (do not invent numbers).

### 5) Risks & Limitations
- List risks tied to assumptions (WACC, growth, terminal g).
- List limitations EXACTLY from context.limitations (no new claims).

### 6) Appendix: Data Notes
- Cache file used (if provided)
- Timestamp and source
- Any warnings flagged by tools
`;

export const generateMemo = async (context: AnalystContext): Promise<string> => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) throw new Error("API Key not found in environment");

  const ai = new GoogleGenAI({ apiKey });

  const prompt = `
    ${MEMO_TEMPLATE_INSTRUCTION}
    
    Here is the DATA CONTEXT you MUST use. Do not use outside data:
    \`\`\`json
    ${JSON.stringify(context, null, 2)}
    \`\`\`
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview', // Using Pro for complex writing task
      contents: prompt,
      config: {
        systemInstruction: SYSTEM_PROMPT,
        temperature: 0.3, // Lower temperature for more factual output
      }
    });

    return response.text || "Error: No text generated.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Error generating memo. Please check API Key or try again.";
  }
};