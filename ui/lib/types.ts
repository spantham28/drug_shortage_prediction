export type TabId = "shortage" | "income" | "synopsis";

export interface ShortageResult {
  shortage_flag: number;
  shortage_probability_pct: number;
  shortage_label: string;
  model_type: string;
  ensemble_members: string[];
}

export interface IncomeResult {
  net_income: number;
  net_income_formatted: string;
  model_name: string;
  target: string;
}

export interface FeatureConfig {
  name: string;
  transform: string;
  median?: number;
  offset?: number;
  type?: string;
  options?: string[];
}

export interface IncomeFeatureSchema {
  features: FeatureConfig[];
  model_name: string;
  target: string;
}

export const SHORTAGE_FIELDS = [
  {
    key: "avg_nadac",
    label: "Drug Acquisition Cost (avg NADAC)",
    hint: "Average NADAC per-unit acquisition price in dollars",
    type: "number" as const,
    step: "0.01",
    placeholder: "e.g. 12.50",
  },
  {
    key: "manufacturer_num",
    label: "Number of Manufacturers",
    hint: "Count of active generic manufacturers",
    type: "number" as const,
    step: "1",
    placeholder: "e.g. 3",
  },
  {
    key: "ingredient_num",
    label: "Number of Ingredients",
    hint: "Active pharmaceutical ingredients in the product",
    type: "number" as const,
    step: "1",
    placeholder: "e.g. 1",
  },
  {
    key: "num_forms",
    label: "Number of Dosage Forms",
    hint: "Distinct dosage forms available (tablet, capsule, etc.)",
    type: "number" as const,
    step: "1",
    placeholder: "e.g. 2",
  },
  {
    key: "liquid_flag",
    label: "Formulation Type",
    hint: "Injectable/liquid products use liquid_flag = 1; solids use 0",
    type: "select" as const,
    options: [
      { value: "1", label: "Injectable / Liquid (liquid_flag = 1)" },
      { value: "0", label: "Solid (liquid_flag = 0)" },
    ],
  },
];
