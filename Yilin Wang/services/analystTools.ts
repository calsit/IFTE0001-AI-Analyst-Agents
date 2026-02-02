import { 
  CapitalStructure, 
  WACCResult, 
  FCFData, 
  DCFResult, 
  UserInputs, 
  ScenarioSensitivity,
  AnalystContext 
} from '../types';

// --- MOCKED DATA (Simulating fetch_yfinance_bundle) ---
// In a real backend, this would call yfinance. Here we use hardcoded recent MSFT data 
// to ensure the application is functional and deterministic as per spec.
const MOCK_MSFT_BUNDLE = {
  market: {
    marketCap: 3150000000000, // ~3.15T
    sharesOutstanding: 7430000000, // ~7.43B
    beta: 0.89,
    price: 425.00
  },
  financials: {
    totalDebt: 106000000000, // ~106B
    cash: 80000000000, // ~80B
    interestExpense: 3000000000, // ~3B
    preTaxIncome: 100000000000, // ~100B
    incomeTax: 18000000000, // ~18B
  },
  cashflow: {
    history: [
      { period: "2023", cfo: 87582000000, capex: 28107000000 },
      { period: "2024 (TTM)", cfo: 110000000000, capex: 45000000000 } // Proxies
    ]
  }
};

// --- TOOL 4.2: extract_capital_structure ---
export const extract_capital_structure = (): CapitalStructure => {
  const { market, financials } = MOCK_MSFT_BUNDLE;
  
  const equity_E = market.marketCap;
  const debt_D = financials.totalDebt;
  const cash = financials.cash;
  const net_debt = debt_D - cash;

  return {
    ticker: "MSFT",
    equity_market_value_E: equity_E,
    debt_book_value_D: debt_D,
    cash_and_equivalents: cash,
    net_debt: net_debt,
    shares_outstanding: market.sharesOutstanding,
    beta: market.beta,
    notes: ["Book debt used as proxy for market value of debt"],
    limitations: ["Reliance on most recent reported balance sheet data"],
    warnings: []
  };
};

// --- TOOL 4.3: estimate_wacc ---
export const estimate_wacc = (
  cap_struct: CapitalStructure, 
  inputs: UserInputs
): WACCResult => {
  const { rf, erp } = inputs;
  const beta = inputs.beta_override ?? cap_struct.beta;
  
  // Cost of Equity (CAPM)
  const ke = rf + (beta * erp);

  // Effective Tax Rate
  // T = Tax / PreTaxIncome
  const t_effective = MOCK_MSFT_BUNDLE.financials.incomeTax / MOCK_MSFT_BUNDLE.financials.preTaxIncome; // ~18%

  // Cost of Debt (Kd)
  // Kd = Interest Expense / Total Debt (Simplified proxy)
  let kd_pre_tax = inputs.kd_override ?? (MOCK_MSFT_BUNDLE.financials.interestExpense / cap_struct.debt_book_value_D);
  
  // Guardrail: If calculated Kd is weird (e.g., < 1%), floor it or warn.
  if (kd_pre_tax < 0.01) kd_pre_tax = 0.04; // Fallback to a reasonable default if data is messy

  const kd_after_tax = kd_pre_tax * (1 - t_effective);

  // Weights
  const V = cap_struct.equity_market_value_E + cap_struct.debt_book_value_D;
  const w_e = cap_struct.equity_market_value_E / V;
  const w_d = cap_struct.debt_book_value_D / V;

  // WACC
  const wacc = (ke * w_e) + (kd_after_tax * w_d);

  return {
    rf,
    erp,
    beta_used: beta,
    cost_of_equity_ke: Number(ke.toFixed(5)),
    tax_rate_effective: Number(t_effective.toFixed(4)),
    cost_of_debt_kd: Number(kd_pre_tax.toFixed(4)),
    weights: { w_e: Number(w_e.toFixed(4)), w_d: Number(w_d.toFixed(4)) },
    wacc: Number(wacc.toFixed(5)),
    warnings: inputs.beta_override ? ["User provided Beta override"] : [],
    limitations: ["Kd estimated via Interest Expense / Book Debt proxy"],
    components: {
      equity_market_value_E: cap_struct.equity_market_value_E,
      debt_book_value_D: cap_struct.debt_book_value_D
    }
  };
};

// --- TOOL 4.4: compute_fcf_proxy ---
export const compute_fcf_proxy = (): FCFData => {
  const history = MOCK_MSFT_BUNDLE.cashflow.history.map(h => ({
    ...h,
    fcf: h.cfo - h.capex
  }));

  // Sort by period descending implies latest first? We'll assume strict order or just take last
  const latest = history[history.length - 1]; // 2024 TTM

  return {
    method: "CFO_minus_Capex_proxy_for_FCFF",
    fcf_history: history,
    fcf0_latest: latest.fcf,
    warnings: [],
    limitations: ["FCFF approx via CFO - Capex; ignores net interest tax shield adjustments"]
  };
};

// --- TOOL 4.5: run_fcff_dcf ---
export const run_fcff_dcf = (
  fcf0: number,
  wacc: number,
  terminal_g: number,
  years: number,
  growth_rate: number,
  net_debt: number,
  shares: number,
  spot_price: number | null
): DCFResult => {
  let pv_forecast = 0;
  let current_fcf = fcf0;

  // 1. Forecast Period PV
  for (let i = 1; i <= years; i++) {
    current_fcf = current_fcf * (1 + growth_rate);
    pv_forecast += current_fcf / Math.pow(1 + wacc, i);
  }

  // 2. Terminal Value PV
  // TV = FCF_final * (1 + g_term) / (WACC - g_term)
  const fcf_final = current_fcf;
  
  // Guardrail: WACC > terminal_g
  const valid_wacc = Math.max(wacc, terminal_g + 0.005); 
  const tv = (fcf_final * (1 + terminal_g)) / (valid_wacc - terminal_g);
  const pv_terminal = tv / Math.pow(1 + wacc, years);

  // 3. Enterprise Value
  const ev = pv_forecast + pv_terminal;

  // 4. Equity Value
  const equity_val = ev - net_debt;

  // 5. Per Share
  const val_per_share = equity_val / shares;

  let upside: number | null = null;
  if (spot_price) {
    upside = (val_per_share - spot_price) / spot_price;
  }

  return {
    assumptions: {
      method: "FCFF_DCF",
      wacc: Number(wacc.toFixed(4)),
      terminal_g,
      forecast_years: years,
      fcf_growth: growth_rate
    },
    pv_forecast,
    pv_terminal,
    enterprise_value_ev: ev,
    equity_value: equity_val,
    value_per_share: val_per_share,
    upside_vs_spot: upside,
    warnings: wacc <= terminal_g ? ["WACC was adjusted to be higher than terminal growth for stability"] : [],
    limitations: ["Standard 2-stage growth model"]
  };
};

// --- TOOL 4.6: run_scenarios_and_sensitivity ---
export const run_scenarios_and_sensitivity = (
  fcf0: number,
  base_wacc: number,
  base_inputs: UserInputs,
  net_debt: number,
  shares: number
): ScenarioSensitivity => {
  const spot = MOCK_MSFT_BUNDLE.market.price;

  // Base
  const base = run_fcff_dcf(
    fcf0, base_wacc, base_inputs.terminal_g, base_inputs.forecast_years, 
    base_inputs.fcf_growth, net_debt, shares, spot
  );

  // Bull: WACC - 1%, Growth + 2%
  const bull = run_fcff_dcf(
    fcf0, Math.max(0.01, base_wacc - 0.01), base_inputs.terminal_g, base_inputs.forecast_years, 
    base_inputs.fcf_growth + 0.02, net_debt, shares, spot
  );

  // Bear: WACC + 1%, Growth - 2%
  const bear = run_fcff_dcf(
    fcf0, base_wacc + 0.01, base_inputs.terminal_g, base_inputs.forecast_years, 
    base_inputs.fcf_growth - 0.02, net_debt, shares, spot
  );

  // Sensitivity Matrix (WACC x Terminal G)
  const wacc_range = [base_wacc - 0.01, base_wacc - 0.005, base_wacc, base_wacc + 0.005, base_wacc + 0.01];
  const g_range = [base_inputs.terminal_g - 0.005, base_inputs.terminal_g, base_inputs.terminal_g + 0.005];

  const matrix: number[][] = [];

  for (const w of wacc_range) {
    const row: number[] = [];
    for (const g of g_range) {
      const res = run_fcff_dcf(
        fcf0, w, g, base_inputs.forecast_years, 
        base_inputs.fcf_growth, net_debt, shares, null
      );
      row.push(res.value_per_share);
    }
    matrix.push(row);
  }

  return {
    scenarios: { base, bull, bear },
    sensitivity: {
      wacc_values: wacc_range,
      g_values: g_range,
      value_per_share_matrix: matrix
    },
    warnings: [],
    limitations: []
  };
};

// --- TOOL 4.7: build_context_json ---
export const build_context_json = (
  inputs: UserInputs,
  cs: CapitalStructure,
  wacc: WACCResult,
  fcf: FCFData,
  sens: ScenarioSensitivity
): AnalystContext => {
  return {
    meta: {
      ticker: "MSFT",
      source: "yfinance (simulated)",
      generated_at_utc: new Date().toISOString(),
      cache_used: true
    },
    inputs: {
      ticker: "MSFT",
      rf: inputs.rf,
      erp: inputs.erp,
      forecast_years: inputs.forecast_years,
      fcf_growth: inputs.fcf_growth,
      terminal_g: inputs.terminal_g
    },
    capital_structure: cs,
    wacc: wacc,
    fcf: fcf,
    valuation: sens.scenarios.base,
    sensitivity: sens,
    limitations: [
      ...cs.limitations,
      ...wacc.limitations,
      ...fcf.limitations,
      ...sens.scenarios.base.limitations
    ],
    warnings: [
      ...cs.warnings,
      ...wacc.warnings
    ]
  };
};
