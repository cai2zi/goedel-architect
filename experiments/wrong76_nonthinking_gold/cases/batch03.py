from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="MATH-500/test/intermediate_algebra/662.json",
        steps=(
            Step("Let us define:\n\n$$\ny = x^2 - 2x\n$$"),
            Step("Let us further simplify by letting:\n\n$$\nz = y - 8 \\Rightarrow y = z + 8\n$$"),
            Step("After simplifying and expanding all terms, we find that the equation reduces to:\n\n$$\n-37440z + 374400 = 0 \\Rightarrow z = 10\n$$"),
            Step("$$\ny = x^2 - 2x \\Rightarrow x^2 - 2x - 18 = 0\n$$"),
            Step("$$\nx = \\frac{2 \\pm \\sqrt{(-2)^2 + 4 \\cdot 18}}{2} = \\frac{2 \\pm \\sqrt{4 + 72}}{2} = \\frac{2 \\pm \\sqrt{76}}{2} = \\frac{2 \\pm 2\\sqrt{19}}{2} = 1 \\pm \\sqrt{19}\n$$"),
            Step("Neither of these values is among the excluded ones, so both are valid."),
        ),
        nodes=(
            definition("y662", "cot_claim", "def y662 (x : ℝ) : ℝ := x^2 - 2 * x", source_steps=(1,)),
            definition("z662", "cot_claim", "def z662 (x : ℝ) : ℝ := y662 x - 8", source_steps=(2,)),
            claim(
                "cot_linear_z_solution662", "lemma cot_linear_z_solution662 (z : ℝ) : -37440 * z + 374400 = 0 ↔ z = 10",
                source_steps=(3,), label="proved", proof="by constructor <;> intro h <;> norm_num at h ⊢ <;> linarith",
                method="linear_arithmetic",
            ),
            claim(
                "back_substitution662", "lemma back_substitution662 (x : ℝ) (h : z662 x = 10) : x^2 - 2 * x - 18 = 0",
                dependencies=("cot_linear_z_solution662",), source_steps=(4,), label="proved",
                proof="by norm_num [z662, y662] at h ⊢; linarith", method="unfold_substitution",
            ),
            target(
                "gold_intermediate_algebra_662",
                "theorem gold_intermediate_algebra_662 (x : ℝ) : x^2 - 2 * x - 18 = 0 ↔ x = 1 - Real.sqrt 19 ∨ x = 1 + Real.sqrt 19",
                dependencies=("back_substitution662",), source_steps=(5, 6), label="proved",
                proof="by\n  have hs : (Real.sqrt 19)^2 = 19 := Real.sq_sqrt (by norm_num)\n  constructor\n  · intro h\n    have hf : (x - (1 - Real.sqrt 19)) * (x - (1 + Real.sqrt 19)) = 0 := by nlinarith\n    rcases mul_eq_zero.mp hf with h | h\n    · left; linarith\n    · right; linarith\n  · rintro (rfl | rfl) <;> nlinarith",
                method="factor_quadratic_using_sqrt_square", statement="The reduced quadratic has exactly the two roots stated by the COT",
            ),
        ),
        fidelity_notes=("The COT's two substitutions and its reduced linear equation are retained; the root theorem proves completeness for the resulting quadratic.",),
    ),
    Case(
        source_id="MATH-500/test/counting_and_probability/430.json",
        steps=(
            Step("For both bags to return to their original contents, the following must be true:\n\n- Alice’s bag: $ \\text{Original} - X + Y = \\text{Original} \\Rightarrow Y = X $\n- Bob’s bag: $ \\text{Original} + X - Y = \\text{Original} \\Rightarrow X = Y $"),
            Step("Thus, the **only way** for both bags to return to their original contents is if **Bob gives back the same color that Alice gave him**."),
            Step("Given that Alice gives a specific color $ X $, Bob has **6 balls** in his bag: the original 5 and the one $ X $."),
            Step("The probability that Bob picks the same color $ X $ is $ \\frac{1}{6} $."),
            Step("$$\n\\boxed{\\dfrac{1}{6}}\n$$"),
        ),
        nodes=(
            definition("bobBallCount430", "problem_grounding", "def bobBallCount430 : ℕ := 6", problem_source_span="Bob starts with five balls and receives Alice's transferred ball"),
            definition("matchingColorMultiplicity430", "formal_bridge", "def matchingColorMultiplicity430 : ℕ := 2", source_steps=(1, 2, 3), statement="Bob has his original X ball and Alice's transferred X ball"),
            claim(
                "return_same_color_condition430", "lemma return_same_color_condition430 : matchingColorMultiplicity430 = 2",
                role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by norm_num [matchingColorMultiplicity430]", method="multiplicity_count",
            ),
            claim(
                "correct_return_probability430", "lemma correct_return_probability430 : (matchingColorMultiplicity430 : ℚ) / bobBallCount430 = 1 / 3",
                dependencies=("return_same_color_condition430",), source_steps=(3, 4), label="proved",
                proof="by norm_num [matchingColorMultiplicity430, bobBallCount430]", method="two_favorable_balls_out_of_six",
            ),
            target(
                "gold_counting_probability_430",
                "theorem gold_counting_probability_430 : (matchingColorMultiplicity430 : ℚ) / bobBallCount430 = 1 / 6",
                dependencies=("correct_return_probability430",), source_steps=(4, 5), label="disproved",
                proof="by norm_num [matchingColorMultiplicity430, bobBallCount430]", method="multiplicity_two_not_one",
                statement="The COT counts the matching color rather than the two matching balls",
            ),
        ),
    ),
    Case(
        source_id="MATH-500/test/precalculus/768.json",
        steps=(
            Step("Therefore:\n\n$$\n\\cot^{-1}\\left(\\frac{1}{x}\\right) = \\theta = \\tan^{-1}(x).\n$$"),
            Step("So the equation becomes:\n\n$$\n\\sin(2 \\tan^{-1}(x)) = \\frac{1}{3}.\n$$"),
            Step("Thus, the equation becomes:\n\n$$\n\\frac{2x}{1 + x^2} = \\frac{1}{3}.\n$$"),
            Step("$$\n6x = 1 + x^2 \\quad \\Rightarrow \\quad x^2 - 6x + 1 = 0.\n$$"),
            Step("$$\nx = \\frac{6 \\pm \\sqrt{(-6)^2 - 4(1)(1)}}{2(1)} = \\frac{6 \\pm \\sqrt{32}}{2} = \\frac{6 \\pm 4\\sqrt{2}}{2} = 3 \\pm 2\\sqrt{2}.\n$$"),
            Step("Indeed, both values satisfy this, so they are valid."),
        ),
        nodes=(
            definition("arccot768", "problem_grounding", "noncomputable def arccot768 (y : ℝ) : ℝ := Real.pi / 2 - Real.arctan y", problem_source_span="cot inverse is taken with values in (0, pi)"),
            claim(
                "cot_inverse_reciprocal768", "lemma cot_inverse_reciprocal768 (x : ℝ) (hx : 0 < x) : arccot768 (1 / x) = Real.arctan x",
                source_steps=(1,), label="proved",
                proof="by\n  rw [show (1 / x : ℝ) = x⁻¹ by ring, arccot768, Real.arctan_inv_of_pos hx]\n  ring",
                method="positive_branch_arctan_inverse",
            ),
            definition("reducedTrig768", "formal_bridge", "noncomputable def reducedTrig768 (x : ℝ) : ℝ := 2 * x / (1 + x^2)", source_steps=(2, 3)),
            claim(
                "rational_to_quadratic768", "lemma rational_to_quadratic768 (x : ℝ) : reducedTrig768 x = 1 / 3 ↔ x^2 - 6*x + 1 = 0",
                source_steps=(3, 4), label="proved",
                proof="by\n  have hp : 0 < 1 + x^2 := by nlinarith [sq_nonneg x]\n  have hn : 1 + x^2 ≠ 0 := ne_of_gt hp\n  constructor\n  · intro h\n    simp only [reducedTrig768] at h\n    field_simp [hn] at h\n    nlinarith\n  · intro h\n    simp only [reducedTrig768]\n    field_simp [hn]\n    nlinarith",
                method="clear_positive_denominator",
            ),
            target(
                "gold_precalculus_768",
                "theorem gold_precalculus_768 (x : ℝ) : x^2 - 6*x + 1 = 0 ↔ x = 3 - 2 * Real.sqrt 2 ∨ x = 3 + 2 * Real.sqrt 2",
                dependencies=("cot_inverse_reciprocal768", "rational_to_quadratic768"), source_steps=(5, 6), label="proved",
                proof="by\n  have hs : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num)\n  constructor\n  · intro h\n    have hf : (x - (3 - 2 * Real.sqrt 2)) * (x - (3 + 2 * Real.sqrt 2)) = 0 := by nlinarith\n    rcases mul_eq_zero.mp hf with h | h\n    · left; linarith\n    · right; linarith\n  · rintro (rfl | rfl) <;> nlinarith",
                method="factor_quadratic_using_sqrt_square",
            ),
        ),
        fidelity_notes=("The positive inverse-cotangent branch used by both positive roots is explicit; the remaining proof is a small rational-to-quadratic bridge.",),
    ),
    Case(
        source_id="MATH-500/test/geometry/817.json",
        steps=(
            Step("Based on the symmetry and the right angle at $ Q $, the most natural configuration is that:\n- One semicircle is tangent to the x-axis (radius $ AQ $), the y-axis (radius $ BQ $), and the main circle."),
            Step("Let the radius of the smaller semicircle be $ r $. Since it is tangent to both axes, its center lies at $ (r, r) $."),
            Step("$$\nr\\sqrt{2} = 14 - r\n$$"),
            Step("$$\nr = \\frac{14}{\\sqrt{2} + 1} \\cdot \\frac{\\sqrt{2} - 1}{\\sqrt{2} - 1} = \\frac{14(\\sqrt{2} - 1)}{(\\sqrt{2} + 1)(\\sqrt{2} - 1)} = \\frac{14(\\sqrt{2} - 1)}{2 - 1} = 14(\\sqrt{2} - 1)\n$$"),
            Step("This is the exact form, and it is **not** a rational number, but it is the simplest exact expression for the radius under the given geometric constraints."),
        ),
        nodes=(
            definition("cotRadius817", "cot_claim", "noncomputable def cotRadius817 : ℝ := 14 * (Real.sqrt 2 - 1)", source_steps=(4,)),
            claim(
                "sqrt_two_square817", "lemma sqrt_two_square817 : (Real.sqrt 2)^2 = 2",
                role="formal_bridge", source_steps=(3, 4), label="proved", proof="by norm_num", method="norm_num_sqrt",
            ),
            claim(
                "cot_tangency_equation817", "lemma cot_tangency_equation817 : cotRadius817 * Real.sqrt 2 = 14 - cotRadius817",
                dependencies=("sqrt_two_square817",), source_steps=(1, 2, 3, 4), label="proved",
                proof="by\n  unfold cotRadius817\n  nlinarith [sqrt_two_square817]", method="solve_assumed_axis_tangency",
            ),
            claim(
                "conditional_radius_unique817",
                "lemma conditional_radius_unique817 (r : ℝ) (hr : 0 ≤ r) (h : r * Real.sqrt 2 = 14 - r) : r = cotRadius817",
                dependencies=("cot_tangency_equation817",), source_steps=(4, 5), label="proved",
                proof="by\n  have hspos : 0 < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)\n  unfold cotRadius817\n  nlinarith [sqrt_two_square817]",
                method="unique_linear_solution_under_cot_geometry",
            ),
            target(
                "gold_geometry_817", "theorem gold_geometry_817 : ∃ q : ℚ, cotRadius817 = (q : ℝ)",
                dependencies=("conditional_radius_unique817",), source_steps=(5,), label="disproved",
                proof="by\n  have hsub : Irrational (Real.sqrt 2 - (1 : ℚ)) := irrational_sqrt_two.sub_ratCast 1\n  have hmul : Irrational ((Real.sqrt 2 - (1 : ℚ)) * (14 : ℚ)) := hsub.mul_ratCast (by norm_num)\n  have hi : Irrational cotRadius817 := by simpa [cotRadius817, mul_comm] using hmul\n  rintro ⟨q, hq⟩\n  exact hi.ne_rat q hq",
                method="irrationality_contradicts_common_fraction",
                statement="The COT explicitly produces an irrational radius even though the requested answer is a common fraction",
            ),
        ),
        fidelity_notes=("Per the agreed diagram policy, the COT's explicit axis-tangent interpretation is used as the geometric source; the irrational/common-fraction tension is preserved in the final source step.",),
    ),
    Case(
        source_id="aime_2024/81",
        steps=(
            Step("The **diameters** of the dodecagon are the lines connecting opposite vertices. There are **6 diameters** in total"),
            Step("Therefore, **any rectangle inscribed in the dodecagon must have its diagonals as diameters** of the dodecagon."),
            Step("Hence, **every pair of diameters** determines a unique rectangle."),
            Step("  $$\n  \\binom{6}{2} = \\frac{6 \\cdot 5}{2} = 15\n  $$"),
            Step("$$\n\\boxed{15}\n$$"),
        ),
        nodes=(
            definition("diameterCount81", "problem_grounding", "def diameterCount81 : ℕ := 6", problem_source_span="twelve vertices form six opposite pairs"),
            definition("interiorRectangleCornerNormSq81", "formal_bridge", "def interiorRectangleCornerNormSq81 : ℚ := (1/2)^2 + (1/2)^2", source_steps=(2, 3), statement="The chords x=±1/2 and y=±1/2 are dodecagon diagonals and bound an allowed rectangle whose corner is (1/2,1/2)"),
            claim(
                "interior_rectangle_not_vertex_inscribed81", "lemma interior_rectangle_not_vertex_inscribed81 : interiorRectangleCornerNormSq81 ≠ 1",
                role="formal_bridge", source_steps=(2, 3), label="proved", proof="by norm_num [interiorRectangleCornerNormSq81]", method="explicit_four_diagonal_rectangle",
                statement="Its corner has squared radius 1/2, so it is not a dodecagon vertex on the unit circumcircle",
            ),
            claim(
                "cot_every_rectangle_vertex_inscribed81", "lemma cot_every_rectangle_vertex_inscribed81 : interiorRectangleCornerNormSq81 = 1",
                dependencies=("interior_rectangle_not_vertex_inscribed81",), source_steps=(2, 3), label="disproved",
                proof="by norm_num [interiorRectangleCornerNormSq81]", method="interior_diagonal_intersection_counterexample",
                statement="The COT silently restricts allowed rectangles to those whose four vertices are dodecagon vertices",
            ),
            claim(
                "choose_diameters81", "lemma choose_diameters81 : Nat.choose diameterCount81 2 = 15",
                source_steps=(1, 4), label="proved",
                proof="by norm_num [diameterCount81, Nat.choose]", method="finite_binomial_count",
            ),
            target(
                "gold_aime_2024_81", "theorem gold_aime_2024_81 : Nat.choose diameterCount81 2 = 15",
                dependencies=("cot_every_rectangle_vertex_inscribed81", "choose_diameters81"), source_steps=(2, 3, 5), label="blocked_by_dependency",
                statement="Fifteen counts only vertex-inscribed rectangles; allowed diagonal-line rectangles also have interior intersection vertices",
            ),
        ),
    ),
    Case(
        source_id="aime_2024/88",
        steps=(
            Step("This means the **major radius** of the torus is $ R = 6 $ (distance from the center of the tube to the center of the torus), and the **minor radius** is $ a = 3 $"),
            Step("The radius of this circle is $ R + a = 6 + 3 = 9 $."),
            Step("The radius of this circle is $ R - a = 6 - 3 = 3 $."),
            Step("$$\nr_i - r_o = 9 - 3 = 6\n$$"),
            Step("Therefore, $ m + n = 6 + 1 = \\boxed{7} $."),
        ),
        nodes=(
            definition("sphereRadius88", "problem_grounding", "def sphereRadius88 : ℚ := 11", problem_source_span="the sphere has radius eleven"),
            definition("majorRadius88", "problem_grounding", "def majorRadius88 : ℚ := 6", problem_source_span="the generating circle center is six from the axis"),
            definition("minorRadius88", "problem_grounding", "def minorRadius88 : ℚ := 3", problem_source_span="the generating circle has radius three"),
            definition("externalContactRadius88", "formal_bridge", "def externalContactRadius88 : ℚ := sphereRadius88 * majorRadius88 / (sphereRadius88 + minorRadius88)", source_steps=(1, 2)),
            definition("internalContactRadius88", "formal_bridge", "def internalContactRadius88 : ℚ := sphereRadius88 * majorRadius88 / (sphereRadius88 - minorRadius88)", source_steps=(1, 3)),
            claim(
                "contact_radii88", "lemma contact_radii88 : externalContactRadius88 = 33 / 7 ∧ internalContactRadius88 = 33 / 4",
                role="formal_bridge", source_steps=(1, 2, 3), label="proved",
                proof="by norm_num [externalContactRadius88, internalContactRadius88, sphereRadius88, majorRadius88, minorRadius88]",
                method="similar_triangles_in_axial_cross_section",
            ),
            target(
                "gold_aime_2024_88", "theorem gold_aime_2024_88 : internalContactRadius88 - externalContactRadius88 = 6",
                dependencies=("contact_radii88",), source_steps=(2, 3, 4, 5), label="disproved",
                proof="by norm_num [externalContactRadius88, internalContactRadius88, sphereRadius88, majorRadius88, minorRadius88]",
                method="sphere_radius_changes_contact_circle",
                statement="The COT incorrectly identifies tangency-circle radii with the torus's radial extrema",
            ),
        ),
    ),
    Case(
        source_id="aime_2024/89",
        steps=(
            Step("This leads to a critical insight: if a row has any chips, then **every column that intersects that row must also have chips of the same color**. This implies that **all rows and columns with chips must be the same color**."),
            Step("All rows in S and columns in T must be **filled with the same color** (say, white or black)."),
            Step("$$\n64 + 64 - 2 = 126\n$$"),
            Step("$$\n\\text{Total valid configurations} = 126 - 1 = \\boxed{125}\n$$"),
        ),
        nodes=(
            definition("SmallGrid89", "formal_bridge", "abbrev SmallGrid89 := Fin 2 → Fin 2 → Option Bool", source_steps=(1,)),
            definition(
                "mixedGrid89", "formal_bridge",
                "def mixedGrid89 : SmallGrid89 := fun i j => if i = j then some (i = 1) else none",
                source_steps=(1, 2), statement="A white chip at (0,0) and a black chip at (1,1)",
            ),
            definition(
                "rowColumnConsistent89", "formal_bridge",
                "def rowColumnConsistent89 (g : SmallGrid89) : Prop := g 0 0 = some false ∧ g 1 1 = some true ∧ g 0 1 = none ∧ g 1 0 = none",
                source_steps=(1,), statement="The two occupied rows and columns each contain only one color",
            ),
            definition(
                "monochrome89", "cot_claim",
                "def monochrome89 (g : SmallGrid89) : Prop := g 0 0 = g 1 1",
                source_steps=(1, 2),
            ),
            claim(
                "mixed_grid_consistent89", "lemma mixed_grid_consistent89 : rowColumnConsistent89 mixedGrid89",
                role="formal_bridge", source_steps=(1, 2), label="proved",
                proof="by simp [rowColumnConsistent89, mixedGrid89]", method="finite_two_by_two_witness",
            ),
            claim(
                "cot_all_chips_same_color89", "lemma cot_all_chips_same_color89 : ∀ g : SmallGrid89, rowColumnConsistent89 g → monochrome89 g",
                dependencies=("mixed_grid_consistent89",), source_steps=(1, 2), label="disproved",
                proof="by\n  push_neg\n  refine ⟨mixedGrid89, ?_, ?_⟩\n  · simp [rowColumnConsistent89, mixedGrid89]\n  · simp [monochrome89, mixedGrid89]", method="two_diagonal_colors_witness",
            ),
            target(
                "gold_aime_2024_89", "theorem gold_aime_2024_89 : (∀ g : SmallGrid89, rowColumnConsistent89 g → monochrome89 g) ∧ 126 - 1 = 125",
                dependencies=("cot_all_chips_same_color89",), source_steps=(3, 4), label="blocked_by_dependency",
                statement="The COT's count 125 depends on the false global-monochromatic reduction",
            ),
        ),
    ),
    Case(
        source_id="aime_2025/8",
        steps=(
            Step("This is the **equation of a circle** centered at $ (25, 20) $ with radius 5."),
            Step("this is the **perpendicular bisector** of the segment joining the points $ (4 + k, 0) $ and $ (k, 3) $."),
            Step("$$\n8x - 6y - (15 + 8k) = 0\n$$"),
            Step("$$\n\\frac{|65 - 8k|}{10} = 5 \\Rightarrow |65 - 8k| = 50\n$$"),
            Step("$$\n65 - 8k = 50 \\quad \\text{or} \\quad 65 - 8k = -50\n$$"),
            Step("The two possible values of $ k $ are $ \\frac{15}{8} $ and $ \\frac{115}{8} $. Their sum is:\n\n$$\n\\frac{15}{8} + \\frac{115}{8} = \\frac{130}{8} = \\frac{65}{4}\n$$"),
            Step("$ m + n = 65 + 4 = \\boxed{69} $."),
        ),
        nodes=(
            definition("tangentEquation8", "formal_bridge", "def tangentEquation8 (k : ℚ) : Prop := |73 - 8*k| = 50", source_steps=(1, 2, 3, 4), statement="Expanding the perpendicular bisector gives 8x-6y-(7+8k)=0, whose distance from (25,20) is |73-8k|/10"),
            claim(
                "solve_tangent_equation8", "lemma solve_tangent_equation8 (k : ℚ) : tangentEquation8 k ↔ k = 23 / 8 ∨ k = 123 / 8",
                source_steps=(4, 5), label="proved",
                proof="by\n  rw [tangentEquation8, abs_eq (by norm_num : (0 : ℚ) ≤ 50)]\n  constructor <;> intro h\n  · rcases h with h | h <;> [left; right] <;> linarith\n  · rcases h with rfl | rfl <;> norm_num",
                method="split_absolute_value",
            ),
            claim(
                "sum_k_values8", "lemma sum_k_values8 : (23 : ℚ) / 8 + 123 / 8 = 73 / 4",
                dependencies=("solve_tangent_equation8",), source_steps=(6,), label="proved", proof="by norm_num", method="norm_num",
            ),
            target(
                "gold_aime_2025_8", "theorem gold_aime_2025_8 : (73 : ℕ) + 4 = 69",
                dependencies=("sum_k_values8",), source_steps=(3, 6, 7), label="disproved", proof="by norm_num", method="correct_perpendicular_bisector_constant",
                statement="The COT uses line constant 15 instead of 7; the corrected tangent values sum to 73/4 and give 77, not 69",
            ),
        ),
    ),
    Case(
        source_id="aime_2025/10",
        steps=(
            Step("Each **block** (A, B, C) is a $3 \\times 3$ grid containing all digits $1$ through $9$. So each block has $9!$ possible arrangements."),
            Step("A more accurate count of the number of such grids is:\n\n$$\n\\left(\\frac{9!}{(3!)^3}\\right)^3\n$$"),
            Step("$\\frac{9!}{(3!)^3} = \\frac{362880}{216} = 1680$"),
            Step("$$\n1680^3 = (2^4 \\cdot 3 \\cdot 5 \\cdot 7)^3 = 2^{12} \\cdot 3^3 \\cdot 5^3 \\cdot 7^3\n$$"),
            Step("$$\n2 \\cdot 12 + 3 \\cdot 3 + 5 \\cdot 3 + 7 \\cdot 3 = 24 + 9 + 15 + 21 = 69\n$$"),
        ),
        nodes=(
            definition(
                "badRows10", "formal_bridge",
                "def badRows10 : List (List ℕ) := [[1,2,3,1,2,3,1,2,3], [4,5,6,4,5,6,4,5,6], [7,8,9,7,8,9,7,8,9]]",
                source_steps=(1, 2), statement="Each consecutive three-column block contains 1 through 9, but rows repeat triples",
            ),
            definition(
                "badBlocksValid10", "formal_bridge",
                "def badBlocksValid10 : Prop := ([1,2,3] ++ [4,5,6] ++ [7,8,9]).Nodup",
                source_steps=(1, 2),
            ),
            definition("badRowsValid10", "formal_bridge", "def badRowsValid10 : Prop := ∀ row ∈ badRows10, row.Nodup", source_steps=(1, 2)),
            claim(
                "independent_blocks_counterexample10", "lemma independent_blocks_counterexample10 : badBlocksValid10 → badRowsValid10",
                source_steps=(1, 2), label="disproved",
                proof="by\n  intro h\n  have hb : badBlocksValid10 := by simp [badBlocksValid10]\n  have hr := h hb\n  have hone := hr [1,2,3,1,2,3,1,2,3] (by simp [badRows10])\n  simp at hone",
                method="repeated_row_triples",
                statement="Choosing a valid digit partition independently in every block need not make global rows permutations",
            ),
            claim(
                "cot_factor_arithmetic10", "lemma cot_factor_arithmetic10 : Nat.factorial 9 / (Nat.factorial 3)^3 = 1680 ∧ 1680^3 = 2^12 * 3^3 * 5^3 * 7^3",
                source_steps=(3, 4), label="proved", proof="by norm_num [Nat.factorial]", method="norm_num",
            ),
            target(
                "gold_aime_2025_10",
                "theorem gold_aime_2025_10 : badRowsValid10 ∧ 2*12 + 3*3 + 5*3 + 7*3 = 69",
                dependencies=("independent_blocks_counterexample10", "cot_factor_arithmetic10"), source_steps=(5,), label="blocked_by_dependency",
                statement="The final factorization answer relies on a block count that does not enforce row validity",
            ),
        ),
    ),
)
