from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="cmimc_2025/35",
        steps=(
            Step("Therefore, $ f(f(x)) $ is also uniformly distributed over $ \\{1, 2, \\ldots, 25\\} $, and thus:"),
            Step("\\mathbb{E}[f(f(x))] = \\frac{1 + 2 + \\cdots + 25}{25} = \\frac{25 \\cdot 26}{2 \\cdot 25} = 13"),
            Step("$$\n\\mathbb{E}[(f(f(x)) - x)^2] = 104 - 2 \\cdot 13 \\cdot x + x^2 = x^2 - 26x + 104\n$$"),
            Step("- $ 5525 - 8450 + 2600 = 1300 $"),
            Step("$$\n\\boxed{1300}\n$$"),
        ),
        nodes=(
            definition("cotMeanSecondIterate35", "cot_claim", "def cotMeanSecondIterate35 : ℚ := 13", source_steps=(1, 2)),
            definition("actualMeanAtOne35", "formal_bridge", "def actualMeanAtOne35 : ℚ := (24 * 325 + 25) / 625", source_steps=(1, 2), statement="Condition f(1)=1 reuses the same random table entry; the other 24 first images use independent entries"),
            claim(
                "actual_mean_at_one35", "lemma actual_mean_at_one35 : actualMeanAtOne35 = 313 / 25",
                role="formal_bridge", source_steps=(1, 2), label="proved", proof="by norm_num [actualMeanAtOne35]", method="condition_on_first_image",
            ),
            claim(
                "cot_second_iterate_uniform35", "lemma cot_second_iterate_uniform35 : actualMeanAtOne35 = cotMeanSecondIterate35",
                dependencies=("actual_mean_at_one35",), source_steps=(1, 2), label="disproved",
                proof="by norm_num [actualMeanAtOne35, cotMeanSecondIterate35]", method="self_composition_dependence",
            ),
            target(
                "gold_cmimc_2025_35", "theorem gold_cmimc_2025_35 : actualMeanAtOne35 = cotMeanSecondIterate35 ∧ 5525 - 8450 + 2600 = 1300",
                dependencies=("cot_second_iterate_uniform35",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The total 1300 uses an incorrect uniform second-iterate distribution",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/37",
        steps=(
            Step("- If $ a_i = 1 $, then:\n  - $ \\binom{1}{0}_3 = 1 $\n  - $ \\binom{1}{1}_3 = 1 $\n  - $ \\binom{1}{2}_3 = 0 $ (since $ x^2 $ is not in the expansion of $ (1 + x + x^2)^1 $)"),
            Step("2024 = 2 \\cdot 3^6 + 2 \\cdot 3^5 + 0 \\cdot 3^4 + 2 \\cdot 3^3 + 2 \\cdot 3^2 + 2 \\cdot 3^1 + 2 \\cdot 3^0"),
            Step("- For $ a_i = 2 $: only $ c_i = 0 $ is valid → 1 choice\n- For $ a_i = 0 $: only $ c_i = 0 $ is valid → 1 choice"),
            Step("$$\n1^7 = 1\n$$"),
            Step("$$\n\\boxed{1}\n$$"),
        ),
        nodes=(
            definition("oneDigitTrinomial37", "formal_bridge", "noncomputable def oneDigitTrinomial37 : Polynomial ℤ := 1 + Polynomial.X + Polynomial.X^2", source_steps=(1,)),
            claim(
                "coefficient_x_squared37", "lemma coefficient_x_squared37 : oneDigitTrinomial37.coeff 2 = 1",
                role="formal_bridge", source_steps=(1,), label="proved", proof="by norm_num [oneDigitTrinomial37, Polynomial.coeff_add, Polynomial.coeff_one, Polynomial.coeff_X, Polynomial.coeff_X_pow]", method="polynomial_coefficient",
            ),
            claim(
                "cot_one_digit_x_squared_zero37", "lemma cot_one_digit_x_squared_zero37 : oneDigitTrinomial37.coeff 2 = 0",
                dependencies=("coefficient_x_squared37",), source_steps=(1,), label="disproved",
                proof="by norm_num [oneDigitTrinomial37, Polynomial.coeff_add, Polynomial.coeff_one, Polynomial.coeff_X, Polynomial.coeff_X_pow]", method="x_squared_is_explicitly_present",
            ),
            target(
                "gold_cmimc_2025_37", "theorem gold_cmimc_2025_37 : oneDigitTrinomial37.coeff 2 = 0 ∧ 1^7 = 1",
                dependencies=("cot_one_digit_x_squared_zero37",), source_steps=(2, 3, 4, 5), label="blocked_by_dependency",
                statement="The Lucas-style digit count begins with a false coefficient of (1+x+x^2)^1",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/4",
        steps=(
            Step("$$\n\\left\\lfloor \\frac{4050}{m} \\right\\rfloor + \\left\\lfloor \\frac{4050}{-m} \\right\\rfloor = \\left\\lfloor \\frac{4050}{m} \\right\\rfloor + \\left\\lfloor -\\frac{4050}{m} \\right\\rfloor\n$$"),
            Step("$$\n1000 - 14 = 986\n$$"),
            Step("$$\n\\left\\lfloor \\frac{4050}{2001} \\right\\rfloor = \\left\\lfloor 2.0239 \\right\\rfloor = 2\n$$"),
            Step("$$\n\\text{Total} = -986 + 2 = -984\n$$"),
            Step("$$\n\\text{Total} = -972 + 2 = \\boxed{-970}\n$$"),
        ),
        nodes=(
            claim("paired_nondisivor_count4", "lemma paired_nondisivor_count4 : (1000 : ℤ) - 14 = 986", source_steps=(1, 2), label="proved", proof="by norm_num", method="integer_arithmetic"),
            claim("unpaired_term4", "lemma unpaired_term4 : (4050 : ℕ) / 2001 = 2", source_steps=(3,), label="proved", proof="by norm_num", method="exact_floor_for_positive_integers"),
            claim(
                "cot_correction_preserves_total4", "lemma cot_correction_preserves_total4 : (-986 : ℤ) + 2 = -970",
                dependencies=("paired_nondisivor_count4", "unpaired_term4"), source_steps=(4, 5), label="disproved",
                proof="by norm_num", method="unexplained_change_986_to_972",
            ),
            target(
                "gold_hmmt_feb_2025_4", "theorem gold_hmmt_feb_2025_4 : (-986 : ℤ) + 2 = -970",
                dependencies=("cot_correction_preserves_total4",), source_steps=(5,), label="blocked_by_dependency",
                statement="The correction replaces the established pair count without justification",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/10",
        steps=(
            Step("$$\n(S - 2)(S + 3)(S^4 - 3S^3 - 26S^2 + 46S + 108) = 0\n$$"),
            Step("This gives us two real roots: $ S = 2 $ and $ S = -3 $."),
            Step("The other roots are complex or extraneous."),
            Step("$$\n\\boxed{-3} \\text{ and } \\boxed{3}\n$$"),
        ),
        nodes=(
            definition("derivedPositiveRoot10", "cot_claim", "def derivedPositiveRoot10 : ℤ := 2", source_steps=(1, 2)),
            definition("boxedPositiveRoot10", "cot_claim", "def boxedPositiveRoot10 : ℤ := 3", source_steps=(4,)),
            claim(
                "cot_positive_root_preserved10", "lemma cot_positive_root_preserved10 : derivedPositiveRoot10 = boxedPositiveRoot10",
                source_steps=(1, 2, 3, 4), label="disproved", proof="by norm_num [derivedPositiveRoot10, boxedPositiveRoot10]",
                method="final_root_changes_two_to_three",
            ),
            target(
                "gold_hmmt_feb_2025_10", "theorem gold_hmmt_feb_2025_10 : derivedPositiveRoot10 = boxedPositiveRoot10 ∧ boxedPositiveRoot10 = 3",
                dependencies=("cot_positive_root_preserved10",), source_steps=(4,), label="blocked_by_dependency",
                statement="The final answer silently changes the derived root 2 to 3",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/13",
        steps=(
            Step("$$\n\\frac{8!}{(2^4 \\cdot 4!)} = \\frac{40320}{16 \\cdot 24} = \\frac{40320}{384} = 105\n$$"),
            Step("$$\n\\frac{105}{4!} = \\frac{105}{24} = 4.375\n$$"),
            Step("This is **not an integer**, which suggests that **some pairings are invalid** due to adjacency constraints. After careful analysis, it turns out that **only one** such valid partition exists"),
            Step("$$\n1 \\text{ (k=4)} + 6 \\text{ (k=3)} + 15 \\text{ (k=2)} + 4 \\text{ (k=1)} + 1 \\text{ (k=0)} = \\boxed{27}\n$$"),
        ),
        nodes=(
            claim("pair_partition_count13", "lemma pair_partition_count13 : Nat.factorial 8 / (2^4 * Nat.factorial 4) = 105", source_steps=(1,), label="proved", proof="by norm_num [Nat.factorial]", method="norm_num"),
            claim(
                "cot_division_yields_one13", "lemma cot_division_yields_one13 : (105 : ℚ) / Nat.factorial 4 = 1",
                dependencies=("pair_partition_count13",), source_steps=(2, 3), label="disproved",
                proof="by norm_num [Nat.factorial]", method="four_point_three_seven_five_not_one",
            ),
            target(
                "gold_hmmt_feb_2025_13", "theorem gold_hmmt_feb_2025_13 : (1 : ℕ) + 6 + 15 + 4 + 1 = 27",
                dependencies=("cot_division_yields_one13",), source_steps=(4,), label="blocked_by_dependency",
                statement="The k=0 contribution is replaced by one after a nonintegral division without a counting argument",
            ),
        ),
    ),
)
