// Type definitions matching the JSON schema provided in the spec

export interface Meta {
  ticker: string;
  source: string;
  generated_at_utc: string;
  cache_used: boolean;
}

export interface UserInputs {
  rf: number;
  erp: number;
  forecast_years: number;
  fcf_growth: number;
  terminal_g: number;
  beta_override?: number | null;
  kd_override?: number | null;
}

export interface CapitalStructure {
  ticker: string;
  equity_market_value_E: number;
  debt_book_value_D: number;
  cash_and_equivalents: number;
  net_debt: number;
  shares_outstanding: number;
  beta: number;
  notes: string[];
  limitations: string[];
  warnings: string[];
}

export interface WACCResult {
  rf: number;
  erp: number;
  beta_used: number;
  cost_of_equity_ke: number;
  tax_rate_effective: number;
  cost_of_debt_kd: number;
  weights: {
    w_e: number;
    w_d: number;
  };
  wacc: number;
  warnings: string[];
  limitations: string[];
  components: {
    equity_market_value_E: number;
    debt_book_value_D: number;
  };
}

export interface FCFData {
  method: string;
  fcf_history: {
    period: string;
    cfo: number;
    capex: number;
    fcf: number;
  }[];
  fcf0_latest: number;
  warnings: string[];
  limitations: string[];
}

export interface DCFResult {
  assumptions: {
    method: string;
    wacc: number;
    terminal_g: number;
    forecast_years: number;
    fcf_growth: number;
  };
  pv_forecast: number;
  pv_terminal: number;
  enterprise_value_ev: number;
  equity_value: number;
  value_per_share: number;
  upside_vs_spot: number | null;
  warnings: string[];
  limitations: string[];
}

export interface ScenarioSensitivity {
  scenarios: {
    base: DCFResult;
    bull: DCFResult;
    bear: DCFResult;
  };
  sensitivity: {
    wacc_values: number[];
    g_values: number[];
    value_per_share_matrix: number[][];
  };
  warnings: string[];
  limitations: string[];
}

export interface AnalystContext {
  meta: Meta;
  inputs: {
    ticker: string;
    rf: number;
    erp: number;
    forecast_years: number;
    fcf_growth: number;
    terminal_g: number;
  };
  capital_structure: CapitalStructure;
  wacc: WACCResult;
  fcf: FCFData;
  valuation: DCFResult; // Base case
  sensitivity: ScenarioSensitivity;
  limitations: string[];
  warnings: string[];
}
