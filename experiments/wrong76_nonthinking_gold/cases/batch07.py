from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="hmmt_feb_2025/18",
        steps=(
            Step("$$\na_i^{(t)} = \\max\\{a_{i - t}, a_{i - t + 1}, \\dots, a_{i + t}\\}\n$$"),
            Step("The **global maximum** of the initial values is in exactly $ m = 201 $ windows (since each position is in $ m $ windows)."),
            Step("$$\n\\text{Number of distinct values} = 1 \\ (\\text{from the global max}) + (n - m) \\ (\\text{from the other windows}) = n - m + 1\n$$"),
            Step("$$\n\\text{Expected number of distinct values} = 2025 - 201 + 1 = \\boxed{1825}\n$$"),
            Step("$$\n\\boxed{1825}\n$$"),
        ),
        nodes=(
            definition("counterWindowMaxima18", "formal_bridge", "def counterWindowMaxima18 : List ℕ := [5,5,4,4,5]", source_steps=(1, 2, 3), statement="For cyclic initial values [5,1,4,2,3] with radius-one windows"),
            claim(
                "counter_distinct_window_values18", "lemma counter_distinct_window_values18 : counterWindowMaxima18.toFinset.card = 2",
                role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by decide", method="five_window_example",
            ),
            claim(
                "cot_each_non_global_window_new18", "lemma cot_each_non_global_window_new18 : counterWindowMaxima18.toFinset.card = 5 - 3 + 1",
                dependencies=("counter_distinct_window_values18",), source_steps=(2, 3), label="disproved",
                proof="by decide", method="overlapping_windows_share_non_global_maxima",
            ),
            target(
                "gold_hmmt_feb_2025_18", "theorem gold_hmmt_feb_2025_18 : counterWindowMaxima18.toFinset.card = 5 - 3 + 1 ∧ 2025 - 201 + 1 = 1825",
                dependencies=("cot_each_non_global_window_new18",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The deterministic n-m+1 count is false even for a five-position cyclic example",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/19",
        steps=(
            Step("A regular hexagon has six sides, with three pairs of opposite edges."),
            Step("$$\nP_{\\text{total}} = 3 \\cdot P\n$$"),
            Step("$$\nP = \\frac{1}{3}\n$$"),
            Step("$$\nP_{\\text{total}} = 3 \\cdot \\frac{1}{3} = 1\n$$"),
            Step("$$\n\\boxed{\\dfrac{1}{2}}\n$$"),
        ),
        nodes=(
            definition("derivedProbability19", "cot_claim", "def derivedProbability19 : ℚ := 3 * (1 / 3)", source_steps=(1, 2, 3, 4)),
            definition("boxedProbability19", "cot_claim", "def boxedProbability19 : ℚ := 1 / 2", source_steps=(5,)),
            claim(
                "derived_probability19", "lemma derived_probability19 : derivedProbability19 = 1",
                source_steps=(1, 2, 3, 4), label="proved", proof="by norm_num [derivedProbability19]", method="norm_num",
            ),
            target(
                "gold_hmmt_feb_2025_19", "theorem gold_hmmt_feb_2025_19 : derivedProbability19 = boxedProbability19",
                dependencies=("derived_probability19",), source_steps=(5,), label="disproved",
                proof="by norm_num [derivedProbability19, boxedProbability19]", method="final_answer_one_to_one_half",
                statement="The boxed one-half contradicts the immediately preceding computed probability one",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/25",
        steps=(
            Step("$$\nAX = 5, \\quad \\angle AXP = 90^\\circ \\Rightarrow \\triangle AXP \\text{ is right-angled at } X\n$$"),
            Step("$$\nXP = \\sqrt{AP^2 - AX^2} = \\sqrt{AP^2 - 25}\n$$"),
            Step("Now, the key is to find $ AP $, which depends on the position of $ P $, which in turn is constrained by the condition $ \\angle BPC = 120^\\circ $."),
            Step("$$\nXP = \\boxed{2}\n$$"),
            Step("$$\nAX^2 + XP^2 = AP^2 \\Rightarrow 5^2 + 2^2 = 25 + 4 = 29 \\Rightarrow AP = \\sqrt{29}\n$$"),
        ),
        nodes=(
            definition("rightTriangleRelation25", "formal_bridge", "def rightTriangleRelation25 (xp apSq : ℚ) : Prop := 25 + xp^2 = apSq", source_steps=(1, 2)),
            claim(
                "cot_right_triangle_alone_forces_two25", "lemma cot_right_triangle_alone_forces_two25 : ∀ xp apSq : ℚ, 0 < xp → rightTriangleRelation25 xp apSq → xp = 2",
                source_steps=(1, 2, 3, 4), label="disproved",
                proof="by\n  push_neg\n  exact ⟨3, 34, by norm_num, by norm_num [rightTriangleRelation25], by norm_num⟩",
                method="xp_three_ap_squared_34",
                statement="The displayed right-triangle relation does not determine XP before the 120-degree condition is used",
            ),
            target(
                "gold_hmmt_feb_2025_25", "theorem gold_hmmt_feb_2025_25 : rightTriangleRelation25 2 29 ∧ (2 : ℚ) = 2",
                dependencies=("cot_right_triangle_alone_forces_two25",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The COT inserts XP=2 without deriving AP or applying the angle condition",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/30",
        steps=(
            Step("$$\n45 + 66 + 63 + 55 + 54 + 77 = 360\n$$"),
            Step("$$\na + b + c = \\frac{360}{2} = 180\n$$"),
            Step("the **distance from the center of the prism to the plane** is equal to the **average of the side lengths** of the hexagon."),
            Step("$$\n\\text{Average} = \\frac{45 + 66 + 63 + 55 + 54 + 77}{6} = \\frac{360}{6} = 60\n$$"),
            Step("$$\n\\boxed{60}\n$$"),
        ),
        nodes=(
            definition("sectionSideSum30", "cot_claim", "def sectionSideSum30 : ℚ := 45 + 66 + 63 + 55 + 54 + 77", source_steps=(1, 2, 4)),
            claim("section_perimeter30", "lemma section_perimeter30 : sectionSideSum30 = 360", source_steps=(1,), label="proved", proof="by norm_num [sectionSideSum30]", method="norm_num"),
            claim(
                "cot_perimeter_uniquely_forces_distance30", "lemma cot_perimeter_uniquely_forces_distance30 : ∀ d : ℚ, sectionSideSum30 = 360 → d = sectionSideSum30 / 6",
                dependencies=("section_perimeter30",), source_steps=(2, 3, 4), label="disproved",
                proof="by\n  push_neg\n  exact ⟨0, by norm_num [sectionSideSum30], by norm_num [sectionSideSum30]⟩",
                method="perimeter_contains_no_plane_offset",
                statement="A section perimeter alone cannot make an arbitrary center-to-plane offset equal its average side length",
            ),
            target(
                "gold_hmmt_feb_2025_30", "theorem gold_hmmt_feb_2025_30 : sectionSideSum30 / 6 = 60",
                dependencies=("cot_perimeter_uniquely_forces_distance30",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The final 60 is only the arithmetic mean of side lengths, not a derived distance",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/6",
        steps=(
            Step("Since $ N = 2025! $, and $ \\phi(M) \\mid M $, and $ M \\mid 2025! $, it follows that $ \\phi(M) \\mid 2025! $, and hence $ \\phi(M) \\mid N $."),
            Step("$$\n2017^N \\equiv 1 \\mod M\n$$"),
            Step("$$\na \\equiv 706 \\mod 2017 \\Rightarrow r = aM = 706M\n$$"),
            Step("$$\n\\frac{r}{N} = \\frac{706M}{2017M} = \\frac{706}{2017}\n$$"),
            Step("$$\n\\boxed{\\dfrac{706}{2017}}\n$$"),
        ),
        nodes=(
            claim(
                "cot_totient_divides_input6", "lemma cot_totient_divides_input6 : ∀ M : ℕ, Nat.totient M ∣ M",
                source_steps=(1,), label="disproved",
                proof="by\n  push_neg\n  refine ⟨3, ?_⟩\n  rw [Nat.totient_prime (by norm_num : Nat.Prime 3)]\n  norm_num",
                method="totient_three_is_two",
                statement="The COT invokes the false general implication phi(M) divides M",
            ),
            target(
                "gold_hmmt_feb_2025_6", "theorem gold_hmmt_feb_2025_6 : (706 : ℚ) / 2017 = 706 / 2017",
                dependencies=("cot_totient_divides_input6",), source_steps=(2, 3, 4, 5), label="blocked_by_dependency",
                statement="The Euler reduction and CRT conclusion depend on the invalid divisibility premise",
            ),
        ),
    ),
)
