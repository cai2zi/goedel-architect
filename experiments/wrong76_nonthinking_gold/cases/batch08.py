from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="aime_2025/20",
        steps=(
            Step("\\angle BAC = 84^\\circ, \\quad \\angle ABC = 60^\\circ, \\quad \\angle ACB = 36^\\circ"),
            Step("- Arc $ \\wideparen{DE} $: subtended by $ \\angle F = 36^\\circ $, so central angle = $ 2 \\times 36^\\circ = 72^\\circ $"),
            Step("$$\n\\wideparen{DE} = 72^\\circ, \\quad \\wideparen{HJ} = 24^\\circ, \\quad \\wideparen{FG} = 70^\\circ\n$$"),
            Step("= 72 + 2 \\cdot 24 + 3 \\cdot 70\n= 72 + 48 + 210 = 330^\\circ"),
            Step("$$\n\\boxed{330}\n$$"),
        ),
        nodes=(
            definition("correctArcFG20", "formal_bridge", "def correctArcFG20 : ℚ := 2 * 36", source_steps=(1, 2, 3), statement="F and G subtend the same 36-degree vertex-angle separation on the medial circle"),
            definition("cotArcFG20", "cot_claim", "def cotArcFG20 : ℚ := 70", source_steps=(3,)),
            claim("correct_arc_fg20", "lemma correct_arc_fg20 : correctArcFG20 = 72", role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by norm_num [correctArcFG20]", method="inscribed_angle_double"),
            claim(
                "cot_arc_fg_value20", "lemma cot_arc_fg_value20 : cotArcFG20 = correctArcFG20",
                dependencies=("correct_arc_fg20",), source_steps=(3,), label="disproved",
                proof="by norm_num [cotArcFG20, correctArcFG20]", method="seventy_vs_seventy_two",
            ),
            target(
                "gold_aime_2025_20", "theorem gold_aime_2025_20 : 72 + 2 * 24 + 3 * cotArcFG20 = 330",
                dependencies=("cot_arc_fg_value20",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The final 330 uses the erroneous 70-degree FG arc",
            ),
        ),
    ),
    Case(
        source_id="brumo_2025/17",
        steps=(
            Step("Assume point $O$ lies on this diagonal, so its coordinates must satisfy $k = h$."),
            Step("$CO = \\sqrt{(x - 2)^2 + (x - 2)^2} = \\sqrt{2(x - 2)^2} = \\sqrt{2}|x - 2| = 2$"),
            Step("$$\n\\sqrt{2}|x - 2| = 2 \\Rightarrow |x - 2| = \\sqrt{2} \\Rightarrow x = 2 \\pm \\sqrt{2}\n$$"),
            Step("If $x = 2 - \\sqrt{2} \\approx 0.586$, then $O = (2, 2)$ would lie outside the square, which is not possible. So we discard this solution."),
            Step("$$\n\\boxed{2 + \\sqrt{2}}\n$$"),
        ),
        nodes=(
            definition("cotSide17", "cot_claim", "noncomputable def cotSide17 : ℝ := 2 + Real.sqrt 2", source_steps=(3, 4, 5)),
            claim(
                "cot_distance_forces_diagonal17", "lemma cot_distance_forces_diagonal17 : ∀ h k : ℚ, h^2 + k^2 = 8 → k = h",
                source_steps=(1,), label="disproved",
                proof="by\n  push_neg\n  exact ⟨14/5, 2/5, by norm_num, by norm_num⟩", method="same_radius_nondiagonal_witness",
                statement="The known distance OA fixes a circle, not the unsupported diagonal equation k=h",
            ),
            claim("sqrt_two_sq17", "lemma sqrt_two_sq17 : (Real.sqrt 2)^2 = 2", role="formal_bridge", source_steps=(2, 3), label="proved", proof="by norm_num", method="sqrt_square"),
            claim(
                "cot_distance_equation17", "lemma cot_distance_equation17 : 2 * (cotSide17 - 2)^2 = 4",
                dependencies=("sqrt_two_sq17",), source_steps=(1, 2, 3), label="proved",
                proof="by unfold cotSide17; nlinarith [sqrt_two_sq17]", method="diagonal_distance",
            ),
            target(
                "gold_brumo_2025_17", "theorem gold_brumo_2025_17 (x : ℝ) (hx : 2 < x) (h : 2 * (x - 2)^2 = 4) : x = cotSide17",
                dependencies=("cot_distance_forces_diagonal17", "cot_distance_equation17"), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The positive branch is valid only after the COT's unsupported diagonal assumption",
            ),
        ),
        fidelity_notes=("The theorem is explicitly conditional on the COT's stated diagonal assumption; it does not silently promote that assumption to the original diagram.",),
    ),
    Case(
        source_id="cmimc_2025/23",
        steps=(
            Step("- $ (x_1, y_1) = (0, 0) $\n- $ (x_2, y_2) = \\left( \\frac{1}{2}, \\frac{\\sqrt{3}}{2} \\right) $\n- $ (x_3, y_3) = (-1, 0) $\n- $ (x_4, y_4) = \\left( \\frac{1}{2}, -\\frac{3\\sqrt{3}}{2} \\right) $\n- $ (x_5, y_5) = (2, 0) $"),
            Step("Sum of $ x_i y_{i+1} = \\frac{3\\sqrt{3}}{2} $"),
            Step("Sum of $ y_i x_{i+1} = -\\frac{\\sqrt{3}}{2} - 3\\sqrt{3} = -\\frac{7\\sqrt{3}}{2} $"),
            Step("$$\n\\left| \\frac{3\\sqrt{3}}{2} - \\left( -\\frac{7\\sqrt{3}}{2} \\right) \\right| = \\left| \\frac{10\\sqrt{3}}{2} \\right| = 5\\sqrt{3}\n$$"),
            Step("$$\n\\text{Area} = \\frac{1}{2} \\cdot 5\\sqrt{3} = \\boxed{\\dfrac{5\\sqrt{3}}{2}}\n$$"),
        ),
        nodes=(
            definition("forwardCoeff23", "cot_claim", "def forwardCoeff23 : ℚ := 3 / 2", source_steps=(1, 2)),
            definition("backwardCoeff23", "cot_claim", "def backwardCoeff23 : ℚ := -7 / 2", source_steps=(1, 3)),
            claim("shoelace_difference23", "lemma shoelace_difference23 : |forwardCoeff23 - backwardCoeff23| = 5", source_steps=(2, 3, 4), label="proved", proof="by norm_num [forwardCoeff23, backwardCoeff23, abs_of_nonneg]", method="rational_shoelace_coefficient"),
            target(
                "gold_cmimc_2025_23", "theorem gold_cmimc_2025_23 : |forwardCoeff23 - backwardCoeff23| / 2 = 5 / 2",
                dependencies=("shoelace_difference23",), source_steps=(5,), label="proved",
                proof="by norm_num [forwardCoeff23, backwardCoeff23, abs_of_nonneg]", method="halve_shoelace_coefficient",
                statement="The area is 5/2 times sqrt(3)",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/25",
        steps=(
            Step("$$\nP_1 = \\left( \\frac{17 - 12\\sqrt{3}}{26}, \\frac{12 + 3\\sqrt{3}}{26} \\right)\n$$"),
            Step("$$\nQ_1 = \\left( \\frac{1}{2}, \\frac{24 + 7\\sqrt{3}}{26} \\right)\n$$"),
            Step("These three points form triangle $ \\triangle GHI $, and we are told that its edges do **not intersect** those of $ \\triangle ABC $."),
            Step("\\text{Distance} = \\sqrt{ \\left( \\frac{12\\sqrt{3}}{13} \\right)^2 + \\left( \\frac{-12}{13} \\right)^2 } = \\sqrt{ \\frac{432 + 144}{169} } = \\frac{24}{13}"),
            Step("$$\n\\boxed{\\dfrac{24}{13}}\n$$"),
        ),
        nodes=(
            definition("p1x25", "cot_claim", "noncomputable def p1x25 : ℝ := (17 - 12 * Real.sqrt 3) / 26", source_steps=(1,)),
            definition("p1y25", "cot_claim", "noncomputable def p1y25 : ℝ := (12 + 3 * Real.sqrt 3) / 26", source_steps=(1,)),
            definition("p2x25", "cot_claim", "noncomputable def p2x25 : ℝ := (17 + 12 * Real.sqrt 3) / 26", source_steps=(1, 2, 3)),
            definition("p2y25", "cot_claim", "noncomputable def p2y25 : ℝ := (3 * Real.sqrt 3 - 12) / 26", source_steps=(1, 2, 3)),
            definition("selectedEdgeHitsB25", "formal_bridge", "def selectedEdgeHitsB25 : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (1-t)*p1x25 + t*p2x25 = 1 ∧ (1-t)*p1y25 + t*p2y25 = 0", source_steps=(1, 2, 3)),
            claim(
                "selected_edge_hits_B25", "lemma selected_edge_hits_B25 : selectedEdgeHitsB25",
                role="formal_bridge", source_steps=(1, 2, 3), label="proved",
                proof="by\n  have hs : (Real.sqrt 3)^2 = 3 := by norm_num\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  refine ⟨1/2 + Real.sqrt 3/8, ?_, ?_, ?_, ?_⟩\n  · nlinarith\n  · nlinarith [sq_nonneg (Real.sqrt 3 - 4)]\n  · simp only [p1x25, p2x25]\n    field_simp\n    nlinarith [hs]\n  · simp only [p1y25, p2y25]\n    field_simp\n    nlinarith [hs]",
                method="explicit_affine_intersection_at_B",
                statement="The selected P1P2 edge passes through B=(1,0), so it intersects an edge of ABC",
            ),
            claim(
                "cot_selected_triangle_avoids_ABC25", "lemma cot_selected_triangle_avoids_ABC25 : ¬ selectedEdgeHitsB25",
                dependencies=("selected_edge_hits_B25",), source_steps=(3,), label="disproved",
                proof="by simpa only [not_not] using selected_edge_hits_B25", method="edge_hits_vertex_B",
            ),
            definition("largestSideSquared25", "formal_bridge", "def largestSideSquared25 : ℚ := (432 + 144) / 169", source_steps=(1, 2, 3, 4)),
            claim("largest_side_squared25", "lemma largest_side_squared25 : largestSideSquared25 = (24 / 13)^2", role="formal_bridge", source_steps=(3, 4), label="proved", proof="by norm_num [largestSideSquared25]", method="squared_distance"),
            target(
                "gold_cmimc_2025_25", "theorem gold_cmimc_2025_25 : largestSideSquared25 = (24 / 13)^2",
                dependencies=("cot_selected_triangle_avoids_ABC25", "largest_side_squared25"), source_steps=(5,), label="blocked_by_dependency",
                statement="The computed side belongs to a triangle whose edge intersects ABC, violating the selection condition",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/34",
        steps=(
            Step("\\text{Area} = \\frac{1}{2} \\left| \\sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \\right|"),
            Step("$$\n\\text{Sum} = 178 \\Rightarrow \\text{Area} = \\frac{1}{2} \\cdot 178 = 89\n$$"),
            Step("$$\n\\text{Sum} = -18 \\Rightarrow \\text{Area} = \\frac{1}{2} \\cdot 18 = 9\n$$"),
            Step("$$\na_{\\text{max}} - a_{\\text{min}} = 89 - 9 = \\boxed{80}\n$$"),
        ),
        nodes=(
            definition("maxDoubleArea34", "cot_claim", "def maxDoubleArea34 : ℤ := 178", source_steps=(1, 2)),
            definition("minSignedDoubleArea34", "cot_claim", "def minSignedDoubleArea34 : ℤ := -18", source_steps=(1, 3)),
            definition("minWitnessCrossSigns34", "formal_bridge", "def minWitnessCrossSigns34 : ℤ × ℤ × ℤ × ℤ := (2, -30, -20, 12)", source_steps=(1, 3), statement="Orientation tests for the nonadjacent edges (10,0)-(3,4) and (6,2)-(0,10) in the claimed minimum ordering"),
            claim(
                "min_witness_edges_cross34", "lemma min_witness_edges_cross34 : minWitnessCrossSigns34.1 * minWitnessCrossSigns34.2.1 < 0 ∧ minWitnessCrossSigns34.2.2.1 * minWitnessCrossSigns34.2.2.2 < 0",
                role="formal_bridge", source_steps=(1, 3), label="proved", proof="by norm_num [minWitnessCrossSigns34]", method="proper_segment_intersection_test",
                statement="Both orientation products are negative, so the two nonadjacent edges cross properly",
            ),
            claim(
                "cot_minimum_witness_is_simple34", "lemma cot_minimum_witness_is_simple34 : ¬ (minWitnessCrossSigns34.1 * minWitnessCrossSigns34.2.1 < 0 ∧ minWitnessCrossSigns34.2.2.1 * minWitnessCrossSigns34.2.2.2 < 0)",
                dependencies=("min_witness_edges_cross34",), source_steps=(3,), label="disproved", proof="by norm_num [minWitnessCrossSigns34]", method="claimed_area9_polygon_self_intersects",
                statement="The ordering used to claim minimum area 9 is self-intersecting and therefore inadmissible",
            ),
            claim("area_extremes34", "lemma area_extremes34 : |maxDoubleArea34| / 2 = 89 ∧ |minSignedDoubleArea34| / 2 = 9", source_steps=(2, 3), label="proved", proof="by norm_num [maxDoubleArea34, minSignedDoubleArea34]", method="integer_shoelace"),
            target("gold_cmimc_2025_34", "theorem gold_cmimc_2025_34 : |maxDoubleArea34| / 2 - |minSignedDoubleArea34| / 2 = 80", dependencies=("cot_minimum_witness_is_simple34", "area_extremes34"), source_steps=(4,), label="blocked_by_dependency", statement="The difference 80 uses an inadmissible self-intersecting minimum witness"),
        ),
        fidelity_notes=("This record verifies the COT's explicit extremal shoelace witnesses and arithmetic; the finite non-self-intersecting enumeration remains the stated source bridge.",),
    ),
    Case(
        source_id="cmimc_2025/39",
        steps=(
            Step("- $ L_1 = \\left(\\frac{418}{5}, \\frac{99}{5}\\right) $\n- $ L_2 = \\left(\\frac{168}{5}, -\\frac{26}{5}\\right) $\n- $ L_3 = \\left(\\frac{138}{5}, \\frac{659}{5}\\right) $"),
            Step("- $ x_1(y_2 - y_3) = \\frac{418}{5} \\cdot (-137) = -\\frac{57266}{5} $"),
            Step("- $ x_2(y_3 - y_1) = \\frac{168}{5} \\cdot 112 = \\frac{18816}{5} $"),
            Step("$$\n\\text{Numerator} = -\\frac{57266}{5} + \\frac{18816}{5} + \\frac{3450}{5} = \\frac{-35000}{5} = -7000\n$$"),
            Step("$$\n\\text{Area} = \\frac{1}{2} \\cdot | -7000 | = \\boxed{3500}\n$$"),
        ),
        nodes=(
            definition("crossAB_C39", "formal_bridge", "def crossAB_C39 : ℤ := 198 * (-336) - (-336) * 448", source_steps=(1,), statement="Signed side test for C relative to directed AB, using coordinates scaled by five"),
            definition("crossAB_X39", "formal_bridge", "def crossAB_X39 : ℤ := 198 * (-138) - (-336) * 534", source_steps=(1,), statement="Signed side test for the COT's square vertex X relative to directed AB"),
            claim(
                "cot_X_same_side_as_triangle39", "lemma cot_X_same_side_as_triangle39 : 0 < crossAB_C39 * crossAB_X39",
                role="formal_bridge", source_steps=(1,), label="proved", proof="by norm_num [crossAB_C39, crossAB_X39]", method="orientation_signs",
                statement="C and the constructed X lie on the same side of AB",
            ),
            claim(
                "cot_AB_square_is_outside39", "lemma cot_AB_square_is_outside39 : crossAB_C39 * crossAB_X39 < 0",
                dependencies=("cot_X_same_side_as_triangle39",), source_steps=(1,), label="disproved", proof="by norm_num [crossAB_C39, crossAB_X39]", method="wrong_rotation_direction",
                statement="An outside square must put X opposite C across AB, but the COT rotates AB in the inward direction",
            ),
            definition("shoelaceNumerator39", "cot_claim", "def shoelaceNumerator39 : ℚ := -57266 / 5 + 18816 / 5 + 3450 / 5", source_steps=(1, 2, 3, 4)),
            claim("shoelace_numerator39", "lemma shoelace_numerator39 : shoelaceNumerator39 = -7000", source_steps=(1, 2, 3, 4), label="proved", proof="by norm_num [shoelaceNumerator39]", method="norm_num"),
            target("gold_cmimc_2025_39", "theorem gold_cmimc_2025_39 : |shoelaceNumerator39| / 2 = 3500", dependencies=("cot_AB_square_is_outside39", "shoelace_numerator39"), source_steps=(5,), label="blocked_by_dependency", statement="The shoelace arithmetic uses a square constructed on the inside of triangle ABC"),
        ),
    ),
)
