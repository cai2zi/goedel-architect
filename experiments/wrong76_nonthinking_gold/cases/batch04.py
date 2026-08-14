from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="aime_2025/11",
        steps=(
            Step("So, we substitute $ x = 34y^2 $ into $ f(x) $, and solve the equation:\n\n$$\ny = f(34y^2)\n$$"),
            Step("Each of these corresponds to a **positive** $ y $-value. Since the parabola is symmetric about the x-axis, each of these positive $ y $-values has a corresponding **negative** counterpart, except $ y = 0 $."),
            Step("$$\n2 \\times 17 + 1 = 35\n$$"),
            Step("$$\n2 \\cdot \\frac{35 + \\sqrt{273}}{68} = \\frac{35 + \\sqrt{273}}{34}\n$$"),
            Step("a + b + c + d = 35 + 1 + 273 + 34 = \\boxed{343}"),
        ),
        nodes=(
            definition("candidateY11", "cot_claim", "def candidateY11 : ℚ := 1 / 34", source_steps=(1, 2)),
            claim(
                "candidate_positive11", "lemma candidate_positive11 : 0 < candidateY11",
                source_steps=(1, 2), label="proved", proof="by norm_num [candidateY11]", method="norm_num",
            ),
            claim(
                "cot_parabola_symmetry_gives_negative11",
                "lemma cot_parabola_symmetry_gives_negative11 : ∀ f : ℚ → ℚ, f (34 * candidateY11^2) = candidateY11 → f (34 * (-candidateY11)^2) = -candidateY11",
                dependencies=("candidate_positive11",), source_steps=(2,), label="disproved",
                proof="by\n  push_neg\n  refine ⟨(fun _ => candidateY11), ?_, ?_⟩\n  · rfl\n  · norm_num [candidateY11]",
                method="same_x_has_one_function_value",
                statement="Reflecting the parabola does not reflect an intersection with the graph of a single-valued f",
            ),
            target(
                "gold_aime_2025_11", "theorem gold_aime_2025_11 : (2 : ℕ) * 17 + 1 = 35 ∧ 35 + 1 + 273 + 34 = 343",
                dependencies=("cot_parabola_symmetry_gives_negative11",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The COT doubles positive y-values using the false symmetry step and obtains 343",
            ),
        ),
    ),
    Case(
        source_id="aime_2025/13",
        steps=(
            Step("A disk is divided into **four quadrants** by two perpendicular diameters."),
            Step("$$\n\\binom{27}{2} = 351\n$$"),
            Step("$$\n\\frac{1}{5} \\times \\frac{1}{3} = \\frac{1}{15}\n$$"),
            Step("$$\n\\mathbb{E}[I] = \\binom{27}{2} \\times \\frac{1}{15} = 351 \\times \\frac{1}{15} = \\frac{117}{5}\n$$"),
            Step("$$\n\\boxed{\\dfrac{257}{5}}\n$$"),
        ),
        nodes=(
            definition("fixedDiameterIntersectionProbability13", "problem_grounding", "def fixedDiameterIntersectionProbability13 : ℚ := 1", problem_source_span="the first two segments are perpendicular diameters and therefore certainly intersect"),
            definition("cotUniformPairProbability13", "cot_claim", "def cotUniformPairProbability13 : ℚ := 1 / 15", source_steps=(2, 3)),
            claim(
                "cot_all_351_pairs_uniform13", "lemma cot_all_351_pairs_uniform13 : fixedDiameterIntersectionProbability13 = cotUniformPairProbability13",
                source_steps=(1, 2, 3), label="disproved",
                proof="by norm_num [fixedDiameterIntersectionProbability13, cotUniformPairProbability13]",
                method="fixed_diameter_pair", statement="The COT assigns the random-chord probability to the fixed diameter pair",
            ),
            target(
                "gold_aime_2025_13", "theorem gold_aime_2025_13 : (351 : ℚ) * cotUniformPairProbability13 = 117 / 5 ∧ 1 + 27 + 117 / 5 = 257 / 5",
                dependencies=("cot_all_351_pairs_uniform13",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The expected-region total uses one probability for non-identically-distributed segment pairs",
            ),
        ),
    ),
    Case(
        source_id="aime_2025/15",
        steps=(
            Step("Thus, the cube of a number is divisible by $3^7$ if and only if $3k \\geq 7$, i.e., $k \\geq \\lceil \\frac{7}{3} \\rceil = 3$. That is, the number must be divisible by $3^3 = 27$."),
            Step("We now count the number of such triples where $a, b, c$ are all divisible by $27$."),
            Step("$$\nN = 27^3 = 19683\n$$"),
            Step("$$\n19683 \\mod 1000 = 683\n$$"),
            Step("$$\n\\boxed{683}\n$$"),
        ),
        nodes=(
            definition("cubeSum15", "formal_bridge", "def cubeSum15 (a b c : ℕ) : ℕ := a^3 + b^3 + c^3", source_steps=(1, 2)),
            claim(
                "counterexample_divisible_sum15", "lemma counterexample_divisible_sum15 : cubeSum15 1 3 314 % 2187 = 0",
                role="formal_bridge", source_steps=(1, 2), label="proved", proof="by norm_num [cubeSum15]", method="explicit_modular_witness",
            ),
            claim(
                "cot_each_entry_divisible27_15",
                "lemma cot_each_entry_divisible27_15 : ∀ a b c : ℕ, cubeSum15 a b c % 2187 = 0 → a % 27 = 0 ∧ b % 27 = 0 ∧ c % 27 = 0",
                dependencies=("counterexample_divisible_sum15",), source_steps=(1, 2), label="disproved",
                proof="by\n  push_neg\n  refine ⟨1, 3, 314, ?_, ?_⟩ <;> norm_num [cubeSum15]",
                method="one_three_314_cancellation",
                statement="Divisibility of a sum of cubes does not force each cube to be divisible separately",
            ),
            target(
                "gold_aime_2025_15", "theorem gold_aime_2025_15 : (27 : ℕ)^3 % 1000 = 683",
                dependencies=("cot_each_entry_divisible27_15",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The remainder 683 follows only after the false restriction to three multiples of 27",
            ),
        ),
    ),
    Case(
        source_id="aime_2025/27",
        steps=(
            Step("$$\nx_2 = x_4 = x_6 = x_8 = x_{10} = a, \\quad x_3 = x_5 = x_7 = x_9 = x_{11} = b\n$$"),
            Step("$$\nP = x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} + x_{11} + s_2 + s_3 + \\ldots + s_{10}\n$$"),
            Step("$$\nP = 5a + 5b + 9s = 5(a + b) + 9s = 20\n$$"),
            Step("$$\n5S + 9\\sqrt{S^2 - 20} = 20\n$$"),
            Step("$$\nm + n + p + q = 9 + 95 + 25 + 14 = \\boxed{143}\n$$"),
        ),
        nodes=(
            definition("correctPerimeter27", "formal_bridge", "def correctPerimeter27 (a b s : ℚ) : ℚ := a + b + 9*s", source_steps=(1, 2)),
            definition("cotPerimeter27", "cot_claim", "def cotPerimeter27 (a b s : ℚ) : ℚ := 5*(a+b) + 9*s", source_steps=(2, 3)),
            claim(
                "cot_counts_radial_diagonals_as_edges27", "lemma cot_counts_radial_diagonals_as_edges27 : ∀ a b s : ℚ, correctPerimeter27 a b s = cotPerimeter27 a b s",
                source_steps=(1, 2, 3), label="disproved",
                proof="by\n  push_neg\n  exact ⟨1, 1, 1, by norm_num [correctPerimeter27, cotPerimeter27]⟩",
                method="polygon_has_only_two_radial_boundary_edges",
                statement="Only A1A2 and A11A1, not all ten radial distances, occur in the polygon perimeter",
            ),
            target(
                "gold_aime_2025_27", "theorem gold_aime_2025_27 : (9 : ℕ) + 95 + 25 + 14 = 143",
                dependencies=("cot_counts_radial_diagonals_as_edges27",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The radical value and 143 are derived from the wrong perimeter equation",
            ),
        ),
    ),
    Case(
        source_id="brumo_2025/3",
        steps=(
            Step("- **Intersection with edge $ HD $** (triangle $ CHD $): occurs at $ t = \\frac{3}{4} $\n- **Intersection with edge $ DI $** (triangle $ DIE $): occurs at $ t = 1 $"),
            Step("4. From $ t = \\frac{6}{7} $ to $ t = 1 $: inside triangle $ DIE $"),
            Step("$$\n\\text{Length of } AI = \\sqrt{(3.5)^2 + \\left(\\frac{\\sqrt{3}}{2}\\right)^2} = \\sqrt{12.25 + 0.75} = \\sqrt{13}\n$$"),
            Step("$$\n\\frac{21 + 14 + 14 + 12}{84} = \\frac{61}{84}\n$$"),
            Step("$$\n\\boxed{\\dfrac{61\\sqrt{13}}{84}}\n$$"),
        ),
        nodes=(
            definition("cotLastBoundary3", "cot_claim", "def cotLastBoundary3 : ℚ := 1", source_steps=(1,)),
            definition("cotLastIntervalStart3", "cot_claim", "def cotLastIntervalStart3 : ℚ := 6 / 7", source_steps=(2,)),
            claim(
                "cot_last_triangle_entry_consistent3", "lemma cot_last_triangle_entry_consistent3 : cotLastBoundary3 = cotLastIntervalStart3",
                source_steps=(1, 2), label="disproved", proof="by norm_num [cotLastBoundary3, cotLastIntervalStart3]",
                method="internal_parameter_mismatch_one_vs_six_sevenths",
            ),
            claim(
                "cot_interval_sum3", "lemma cot_interval_sum3 : (1 : ℚ)/4 + (1/2-1/3) + (2/3-1/2) + (1-6/7) = 61/84",
                source_steps=(3, 4), label="proved", proof="by norm_num", method="norm_num",
            ),
            target(
                "gold_brumo_2025_3", "theorem gold_brumo_2025_3 : cotLastBoundary3 = cotLastIntervalStart3 ∧ (1 : ℚ)/4 + (1/2-1/3) + (2/3-1/2) + (1-6/7) = 61/84",
                dependencies=("cot_last_triangle_entry_consistent3", "cot_interval_sum3"), source_steps=(5,), label="blocked_by_dependency",
                statement="The final length uses a last interval inconsistent with the preceding intersection list",
            ),
        ),
    ),
    Case(
        source_id="brumo_2025/6",
        steps=(
            Step("there are **nine 9s** in a row, and Joshua can insert a multiplication sign between any two adjacent digits, effectively splitting the row into two parts."),
            Step("1. $9 \\times 999999999 = 8,999,999,991$"),
            Step("S &= 8,999,999,991 + 899,999,901 + 998,999,001 + 999,890,001"),
            Step("$$\n1 + 1 + 8 + 9 + 8 + 8 + 8 + 8 + 9 + 4 = 64\n$$"),
            Step("$$\n\\boxed{64}\n$$"),
        ),
        nodes=(
            definition("correctFirstRightFactor6", "problem_grounding", "def correctFirstRightFactor6 : ℕ := 99999999", problem_source_span="placing the multiplication sign after the first of nine 9 cards leaves eight 9s"),
            definition("cotFirstRightFactor6", "cot_claim", "def cotFirstRightFactor6 : ℕ := 999999999", source_steps=(2,)),
            claim(
                "cot_first_split_uses_ten_cards6", "lemma cot_first_split_uses_ten_cards6 : correctFirstRightFactor6 = cotFirstRightFactor6",
                source_steps=(1, 2), label="disproved", proof="by norm_num [correctFirstRightFactor6, cotFirstRightFactor6]",
                method="digit_count_eight_vs_nine",
            ),
            target(
                "gold_brumo_2025_6", "theorem gold_brumo_2025_6 : 1 + 1 + 8 + 9 + 8 + 8 + 8 + 8 + 9 + 4 = 64",
                dependencies=("cot_first_split_uses_ten_cards6",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The digit sum is taken from a total whose first product uses one extra card",
            ),
        ),
    ),
    Case(
        source_id="brumo_2025/12",
        steps=(
            Step("$$\ni - j \\equiv -1, 0, \\text{ or } 1 \\pmod{10}\n$$"),
            Step("$$\na(n) = a(n-1) + a(n-2) - (-1)^n\n$$"),
            Step("- $ a(3) = a(2) + a(1) - (-1)^3 = 2 + 1 - (-1) = 4 $"),
            Step("- $ a(10) = a(9) + a(8) - (-1)^{10} = 64 + 38 - 1 = 101 $"),
            Step("$$\n\\boxed{101}\n$$"),
        ),
        nodes=(
            definition("validPermutationCount3_12", "formal_bridge", "def validPermutationCount3_12 : ℕ := Nat.factorial 3", source_steps=(1, 2, 3), statement="Modulo three, -1, 0, and 1 cover every residue, so every permutation is allowed"),
            claim(
                "three_cycle_base_count12", "lemma three_cycle_base_count12 : validPermutationCount3_12 = 6",
                role="formal_bridge", source_steps=(1, 3), label="proved", proof="by norm_num [validPermutationCount3_12, Nat.factorial]", method="enumerate_three_permutations",
            ),
            claim(
                "cot_recurrence_base_a3_12", "lemma cot_recurrence_base_a3_12 : validPermutationCount3_12 = 4",
                dependencies=("three_cycle_base_count12",), source_steps=(2, 3), label="disproved",
                proof="by norm_num [validPermutationCount3_12, Nat.factorial]", method="cyclic_boundary_changes_base_case",
            ),
            target(
                "gold_brumo_2025_12", "theorem gold_brumo_2025_12 : validPermutationCount3_12 = 4 ∧ 64 + 38 - 1 = 101",
                dependencies=("cot_recurrence_base_a3_12",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The recurrence table starts from a false cyclic n=3 count",
            ),
        ),
    ),
    Case(
        source_id="brumo_2025/22",
        steps=(
            Step("$$\n\\text{Total sum} = \\frac{9 \\cdot 10}{2} = 45\n$$"),
            Step("$$\n\\frac{45}{3} = 15\n$$"),
            Step("- Rows: 4+9+2 = 15, 8+1+6 = 15, 3+5+7 = 15\n- Columns: 4+8+3 = 15, 9+1+5 = 15, 2+6+7 = 15"),
            Step("the number of such valid arrangements is **greater than 8**."),
            Step("$$\n\\boxed{8}\n$$"),
        ),
        nodes=(
            definition("cotFinalCount22", "cot_claim", "def cotFinalCount22 : ℕ := 8", source_steps=(5,)),
            claim(
                "common_line_sum22", "lemma common_line_sum22 : (45 : ℕ) / 3 = 15",
                source_steps=(1, 2), label="proved", proof="by norm_num", method="total_sum",
            ),
            claim(
                "displayed_rows_columns22", "lemma displayed_rows_columns22 : 4+9+2=15 ∧ 8+1+6=15 ∧ 3+5+7=15 ∧ 4+8+3=15 ∧ 9+1+5=15 ∧ 2+6+7=15",
                source_steps=(3,), label="proved", proof="by norm_num", method="norm_num",
            ),
            claim(
                "cot_greater_than_final_count22", "lemma cot_greater_than_final_count22 : 8 < cotFinalCount22",
                dependencies=("displayed_rows_columns22",), source_steps=(4, 5), label="disproved",
                proof="by norm_num [cotFinalCount22]", method="explicit_internal_contradiction",
                statement="The COT proves there are more than eight valid grids and then boxes eight",
            ),
            target(
                "gold_brumo_2025_22", "theorem gold_brumo_2025_22 : 8 < cotFinalCount22 ∧ cotFinalCount22 = 8",
                dependencies=("common_line_sum22", "cot_greater_than_final_count22"), source_steps=(5,), label="blocked_by_dependency",
                statement="The final count eight contradicts the COT's own non-magic witness and strict lower bound",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/5",
        steps=(
            Step("Let $ N = 999,999 $, the largest number less than 1,000,000."),
            Step("$$\n\\left\\lfloor \\frac{999,999}{77} \\right\\rfloor = 12,999\n$$"),
            Step("This is done by leveraging the **uniform distribution** of digits in the arithmetic progression of multiples of 77."),
            Step("(1 + 3 + 5 + 7 + 9) \\times 1,300 = 25 \\times 1,300 = 32,500"),
            Step("$$\n32,500 + 32,500 + 32,491 + 32,466 + 32,175 + 29,250 = \\boxed{191382}\n$$"),
        ),
        nodes=(
            claim(
                "actual_multiple_count5", "lemma actual_multiple_count5 : (999999 : ℕ) / 77 = 12987",
                role="formal_bridge", source_steps=(1, 2), label="proved", proof="by norm_num", method="exact_division",
            ),
            claim(
                "cot_multiple_count5", "lemma cot_multiple_count5 : (999999 : ℕ) / 77 = 12999",
                dependencies=("actual_multiple_count5",), source_steps=(2,), label="disproved", proof="by norm_num", method="incorrect_floor",
            ),
            target(
                "gold_cmimc_2025_5", "theorem gold_cmimc_2025_5 : (25 : ℕ) * 1300 = 32500 ∧ 32500 + 32500 + 32491 + 32466 + 32175 + 29250 = 191382",
                dependencies=("cot_multiple_count5",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="Every positional digit count is based on the wrong number of multiples",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/7",
        steps=(
            Step("- $ a_1 = 1 $\n- $ a_2 = 12 $\n- $ a_3 = 123 $\n- $ a_4 = 1234 $"),
            Step("Specifically, $ \\nu_3(a_n) $ increases by 1 at $ n = 9, 18, 27, \\ldots $, and these values are **multiples of 9**."),
            Step("$$\n\\left\\lfloor \\frac{810}{9} \\right\\rfloor = 90\n$$"),
            Step("This suggests that the total sum of $ \\nu_3(a_n) $ from $ n = 1 $ to $ n = 810 $ is the **sum of all integers from 1 to 90**"),
            Step("$$\n\\sum_{n=1}^{810} \\nu_3(a_n) = \\sum_{k=1}^{90} k = \\frac{90 \\cdot 91}{2} = 4095\n$$"),
        ),
        nodes=(
            definition("a9_7", "formal_bridge", "def a9_7 : ℕ := 123456789", source_steps=(1, 2)),
            claim(
                "a9_divisible_by_nine7", "lemma a9_divisible_by_nine7 : a9_7 % 9 = 0",
                role="formal_bridge", source_steps=(1, 2), label="proved", proof="by norm_num [a9_7]", method="exact_remainder",
            ),
            claim(
                "cot_a9_valuation_one7", "lemma cot_a9_valuation_one7 : a9_7 % 9 ≠ 0",
                dependencies=("a9_divisible_by_nine7",), source_steps=(2,), label="disproved", proof="by norm_num [a9_7]",
                method="a9_has_at_least_two_factors_of_three",
                statement="The claimed first increment treats nu3(a9) as one, but a9 is divisible by nine",
            ),
            target(
                "gold_cmimc_2025_7", "theorem gold_cmimc_2025_7 : (90 : ℕ) * 91 / 2 = 4095",
                dependencies=("cot_a9_valuation_one7",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The triangular sum relies on the false valuation pattern",
            ),
        ),
    ),
)
