from __future__ import annotations


def proof(record_id: str, node_name: str, stage: str, proof_body: str, reasoning: str,
          prefix_code: str = ""):
    return {
        "record_id": record_id,
        "node_name": node_name,
        "stage": stage,
        "proof_body": proof_body,
        "reasoning": reasoning,
        "prefix_code": prefix_code,
    }


def rectangle_count(r):
    x1, y1, x2, y2 = r
    return sum(
        x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1
        for a1 in range(6) for b1 in range(6)
        for a2 in range(a1 + 1, 6) for b2 in range(b1 + 1, 6)
    )


RECTANGLES = [
    (x1, y1, x2, y2)
    for x1 in range(6) for y1 in range(6)
    for x2 in range(x1 + 1, 6) for y2 in range(y1 + 1, 6)
]

RECTANGLE_HELPER_NAMES = ["codex_rect_" + "_".join(map(str, r)) for r in RECTANGLES]
RECTANGLE_MIRROR = """
def codex_allRects := Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.range 6)))
def codex_validB (r : ℕ × ℕ × ℕ × ℕ) : Bool := decide (r.1 < r.2.2.1) && decide (r.2.1 < r.2.2.2) && decide (r.2.2.1 ≤ 5) && decide (r.2.2.2 ≤ 5)
def codex_nonoverlapB (r s : ℕ × ℕ × ℕ × ℕ) : Bool := decide (r.2.2.1 < s.1) || decide (s.2.2.1 < r.1) || decide (r.2.2.2 < s.2.1) || decide (s.2.2.2 < r.2.1)
def codex_rectCountB (r : ℕ × ℕ × ℕ × ℕ) : ℕ := if codex_validB r = true then (codex_allRects.filter (fun s => codex_validB s = true ∧ codex_nonoverlapB r s = true)).card else 0
def codex_sumB : ℕ := (codex_allRects.filter (fun r => codex_validB r = true)).sum codex_rectCountB
lemma codex_validB_eq (r) : codex_validB r = true ↔ is_valid_rectangle r := by simp [codex_validB, is_valid_rectangle]; aesop
lemma codex_nonoverlapB_eq (r s) : codex_nonoverlapB r s = true ↔ rectangles_are_nonoverlapping r s := by simp [codex_nonoverlapB, rectangles_are_nonoverlapping]; aesop
lemma codex_count_eq_B (r) : count_nonoverlapping_with r = codex_rectCountB r := by
  classical
  simp only [count_nonoverlapping_with, codex_rectCountB, codex_allRects]
  by_cases h : is_valid_rectangle r
  · rw [dif_pos h, if_pos ((codex_validB_eq r).mpr h)]
    congr 1
    ext s
    simp [codex_validB_eq, codex_nonoverlapB_eq]
  · rw [dif_neg h, if_neg (by simpa [codex_validB_eq] using h)]
lemma codex_sum_eq_B : sum_nonoverlapping_S = codex_sumB := by
  classical
  simp only [sum_nonoverlapping_S, codex_sumB, codex_allRects]
  apply Finset.sum_congr
  · ext r
    simp [codex_validB_eq]
  · intro r hr
    exact codex_count_eq_B r
"""
RECTANGLE_PREFIX = RECTANGLE_MIRROR + "\n\n" + "\n\n".join(
    f"lemma {name} : codex_rectCountB {r} = {rectangle_count(r)} := by\n"
    "  set_option maxRecDepth 100000 in decide"
    for name, r in zip(RECTANGLE_HELPER_NAMES, RECTANGLES, strict=True)
)

HMMT10_PREFIX = """
noncomputable def codex_q : Polynomial ℂ :=
  Polynomial.X^6 + Polynomial.X^5 - 17*Polynomial.X^4 - 11*Polynomial.X^3 +
    91*Polynomial.X^2 + 25*Polynomial.X - 149
lemma codex_q_has_root : ∃ z : ℂ, codex_q.IsRoot z := by
  apply Complex.exists_root
  have hc : codex_q.coeff 6 ≠ 0 := by norm_num [codex_q, Polynomial.coeff_X_pow, Polynomial.coeff_X]
  exact lt_of_lt_of_le (by norm_num) (Polynomial.le_degree_of_ne_zero hc)
"""


MANUAL_PROOFS = [
    proof(
        "robustpa_MATH_500_test_prealgebra_874_json",
        "same_side_interior_angles",
        "negative",
        "by\n  intro h\n  have h0 := h 0\n  norm_num [angle_at_T, angle_at_R] at h0",
        "counterexample_x_zero",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_880_json",
        "horizontal_dist_is_10",
        "positive",
        "by\n  norm_num [horizontal_distance, fly_unfolded, gecko_unfolded, fly_x, gecko_x]",
        "unfold_coordinate_definitions_and_normalize",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_880_json",
        "vertical_dist_is_4",
        "positive",
        "by\n  norm_num [vertical_distance, fly_unfolded, gecko_unfolded, fly_z, gecko_z, room_width]",
        "unfold_coordinate_definitions_and_normalize",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_prealgebra_1865_json", name, "positive",
            f"by\n  norm_num [num_riders, {pct}, {survey}]",
            "unfold_decimal_percentage_calculation",
        )
        for name, pct, survey in [
            ("num_male_9th", "male_pct_9th", "survey_males_per_grade"),
            ("num_female_9th", "female_pct_9th", "survey_females_per_grade"),
            ("num_male_10th", "male_pct_10th", "survey_males_per_grade"),
            ("num_female_10th", "female_pct_10th", "survey_females_per_grade"),
            ("num_male_11th", "male_pct_11th", "survey_males_per_grade"),
            ("num_female_11th", "female_pct_11th", "survey_females_per_grade"),
            ("num_male_12th", "male_pct_12th", "survey_males_per_grade"),
            ("num_female_12th", "female_pct_12th", "survey_females_per_grade"),
        ]
    ],
    proof(
        "robustpa_aime_2025_27", "area_product_constraint", "positive",
        "by\n  intro x y hx hy h\n  norm_num [sin_theta] at h ⊢\n  linarith",
        "substitute_sine_and_solve_linear_product_equation",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_817_json", "distance_formula", "positive",
        "by\n  simp [smaller_semicircle_center]\n  rw [show r ^ 2 + r ^ 2 = 2 * r ^ 2 by ring]\n  rw [Real.sqrt_mul (by positivity : (0 : ℝ) ≤ 2)]\n  rw [Real.sqrt_sq_eq_abs, abs_of_pos hr]\n  ring",
        "distance_formula_and_positive_radius",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_817_json", "solve_tangency_equation", "positive",
        "by\n  intro r\n  simp only [main_circle_radius]\n  have hs : (Real.sqrt 2) ^ 2 = (2 : ℝ) := by norm_num\n  have hn : Real.sqrt 2 + 1 ≠ 0 := by positivity\n  constructor\n  · intro h\n    field_simp\n    nlinarith\n  · intro h\n    field_simp at h\n    nlinarith",
        "solve_linear_equation_with_nonzero_sqrt_denominator",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_817_json", "rationalize_denominator", "positive",
        "by\n  simp only [main_circle_radius]\n  have hs : (Real.sqrt 2) ^ 2 = (2 : ℝ) := by norm_num\n  have hn : Real.sqrt 2 + 1 ≠ 0 := by positivity\n  field_simp\n  nlinarith",
        "rationalize_using_sqrt_two_square",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_880_json", "horizontal_dist_sq_is_100", "positive",
        "by\n  rw [horizontal_distance_sq, horizontal_dist_is_10]\n  norm_num",
        "rewrite_verified_distance_and_normalize",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_880_json", "vertical_dist_sq_is_16", "positive",
        "by\n  rw [vertical_distance_sq, vertical_dist_is_4]\n  norm_num",
        "rewrite_verified_distance_and_normalize",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_880_json", "sum_sq_is_116", "positive",
        "by\n  rw [sum_squared_distances, horizontal_dist_sq_is_100, vertical_dist_sq_is_16]\n  norm_num",
        "sum_verified_squared_distances",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_880_json", "sqrt_116_is_2_sqrt_29", "positive",
        "by\n  rw [show (116 : ℝ) = 4 * 29 by norm_num]\n  rw [Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 4)]\n  norm_num",
        "factor_radicand_and_use_sqrt_product",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_880_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_geometry_880_json",
        "positive",
        "by\n  rw [shortest_path_distance, sum_sq_is_116, sqrt_116_is_2_sqrt_29]",
        "rewrite_distance_chain",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_prealgebra_1865_json", name, "positive",
            f"by\n  norm_num [{pct}, multiplier_135pct, num_riders, {female_pct}, {survey}]",
            "unfold_decimal_percentage_calculation",
        )
        for name, pct, female_pct, survey in [
            ("pct135_female_9th", "num_female_9th", "female_pct_9th", "survey_females_per_grade"),
            ("pct135_female_10th", "num_female_10th", "female_pct_10th", "survey_females_per_grade"),
            ("pct135_female_11th", "num_female_11th", "female_pct_11th", "survey_females_per_grade"),
            ("pct135_female_12th", "num_female_12th", "female_pct_12th", "survey_females_per_grade"),
        ]
    ],
    *[
        proof(
            "robustpa_MATH_500_test_prealgebra_1865_json", name, "positive",
            f"by\n  norm_num [{diff_def}, multiplier_135pct, num_riders, {male_pct}, {female_pct}, {male_survey}, {female_survey}, abs_of_nonneg]",
            "unfold_absolute_decimal_difference",
        )
        for name, diff_def, male_pct, female_pct, male_survey, female_survey in [
            ("diff_9th_value", "diff_9th_grade", "male_pct_9th", "female_pct_9th", "survey_males_per_grade", "survey_females_per_grade"),
            ("diff_10th_value", "diff_10th_grade", "male_pct_10th", "female_pct_10th", "survey_males_per_grade", "survey_females_per_grade"),
            ("diff_11th_value", "diff_11th_grade", "male_pct_11th", "female_pct_11th", "survey_males_per_grade", "survey_females_per_grade"),
            ("diff_12th_value", "diff_12th_grade", "male_pct_12th", "female_pct_12th", "survey_males_per_grade", "survey_females_per_grade"),
        ]
    ],
    *[
        proof(
            "robustpa_MATH_500_test_prealgebra_1865_json", name, "positive",
            f"by\n  rw [diff_12th_value, {other}]\n  norm_num",
            "compare_verified_decimal_differences",
        )
        for name, other in [
            ("diff_12th_lt_9th", "diff_9th_value"),
            ("diff_12th_lt_10th", "diff_10th_value"),
            ("diff_12th_lt_11th", "diff_11th_value"),
        ]
    ],
    proof(
        "robustpa_MATH_500_test_prealgebra_1865_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_prealgebra_1865_json",
        "positive",
        "by\n  constructor\n  · exact le_of_lt diff_12th_lt_9th\n  constructor\n  · exact le_of_lt diff_12th_lt_10th\n  constructor\n  · exact le_of_lt diff_12th_lt_11th\n  · exact diff_12th_value",
        "assemble_verified_grade_comparisons",
    ),
    proof(
        "robustpa_MATH_500_test_precalculus_1056_json", "sum_scaled", "positive",
        "by\n  simp only [d1_sq, d2_sq, d3_sq]\n  ring",
        "expand_squared_distances_and_ring",
    ),
    proof(
        "robustpa_MATH_500_test_counting_and_probability_765_json", "inclusion_exclusion_eq", "negative",
        "by\n  intro h\n  have h0 := h 0\n  norm_num [at_least_one, total_students, no_subjects, calc_count, chem_count, phys_calc_total, phys_calc_only, calc_chem, phys_chem, all_three] at h0",
        "counterexample_physics_count_zero",
    ),
    proof(
        "robustpa_MATH_500_test_counting_and_probability_765_json", "verification_sum", "negative",
        "by\n  intro h\n  have h0 := h 0\n  norm_num [only_calculus_count, only_physics_count, only_chemistry_count, only_phys_calc_count, only_calc_chem_count, only_phys_chem_count, all_three, at_least_one, total_students, no_subjects, calc_count, phys_calc_total, phys_calc_only, calc_chem, phys_chem, chem_count] at h0",
        "counterexample_physics_count_zero",
    ),
    proof(
        "robustpa_hmmt_feb_2025_16", "total_pairs_is_25200", "positive",
        "by\n  norm_num [total_rectangle_pairs_count, total_rectangles_count, grid_line_count]",
        "unfold_rectangle_pair_count",
    ),
    proof(
        "robustpa_MATH_500_test_counting_and_probability_430_json", "total_probability", "positive",
        "by\n  norm_num [num_colors]",
        "normalize_rational_probability",
    ),
    proof(
        "robustpa_hmmt_feb_2025_16", "total_pairs_is_25200", "positive",
        "by\n  decide",
        "kernel_checked_computation_of_finite_choose",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_prealgebra_1139_json", name, "positive",
            "by\n  decide",
            "kernel_checked_finite_evaluation",
        )
        for name in [
            "value_121_obtainable", "value_144_obtainable", "value_126_obtainable",
            "value_160_obtainable", "value_150_obtainable", "value_180_obtainable",
            "value_71_obtainable", "value_101_obtainable", "value_51_obtainable",
            "value_35_obtainable", "value_47_obtainable", "value_84_obtainable",
            "all_values_distinct", "obtainable_equals_claimed",
            "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_prealgebra_1139_json",
        ]
    ],
    proof(
        "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "positive",
        "by\n  decide",
        "kernel_checked_finite_path_enumeration",
    ),
    proof(
        "robustpa_hmmt_feb_2025_14",
        "robustpa_qwen3_8b_math_verify_hmmt_feb_2025_robustpa_hmmt_feb_2025_14",
        "positive",
        "by\n  exact dfs_count_lemma",
        "reuse_verified_path_count",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  decide",
        "kernel_checked_finite_position_count",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "total_removals_value", "positive",
        "by\n  decide",
        "kernel_checked_finite_choose",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12",
        "robustpa_qwen3_8b_math_verify_hmmt_feb_2025_robustpa_hmmt_feb_2025_12",
        "positive",
        "by\n  decide",
        "kernel_checked_finite_disconnect_enumeration",
    ),
    proof(
        "robustpa_brumo_2025_22", "row_sum_must_be_15", "positive",
        "by\n  rcases h with ⟨_, hr, hc⟩\n  exact ⟨hr, hc⟩",
        "valid_grid_definition_contains_row_and_column_conditions",
    ),
    proof(
        "robustpa_brumo_2025_22", "counterexample_exists", "positive",
        "by\n  decide",
        "kernel_checked_counterexample_grid",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_prealgebra_1139_json", name, "positive",
            f"by\n  refine ⟨⟨{index}, by norm_num⟩, ?_⟩\n  norm_num [evaluate_with_paren]",
            "explicit_parenthesization_witness",
        )
        for name, index in [
            ("value_121_obtainable", 0), ("value_144_obtainable", 1),
            ("value_126_obtainable", 2), ("value_160_obtainable", 3),
            ("value_150_obtainable", 4), ("value_180_obtainable", 5),
            ("value_71_obtainable", 6), ("value_101_obtainable", 7),
            ("value_51_obtainable", 8), ("value_35_obtainable", 9),
            ("value_47_obtainable", 10), ("value_84_obtainable", 11),
        ]
    ],
    proof(
        "robustpa_MATH_500_test_prealgebra_1139_json", "all_values_distinct", "positive",
        "by\n  norm_num [claimed_values, Set.ncard_insert_of_not_mem]",
        "normalize_finite_literal_set_cardinality",
    ),
    proof(
        "robustpa_hmmt_feb_2025_16", "total_pairs_is_25200", "positive",
        "by\n  norm_num [total_rectangle_pairs_count, total_rectangles_count, grid_line_count, Nat.choose]",
        "unfold_choose_recursion_and_normalize",
    ),
    proof(
        "robustpa_brumo_2025_22", "counterexample_exists", "positive",
        "by\n  norm_num [counterexample_grid, is_valid_grid, is_magic_square, is_permutation_1_to_9, rows_sum_to_15, cols_sum_to_15]",
        "unfold_and_validate_explicit_counterexample",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_817_json", "tangency_equation", "positive",
        "by\n  rw [distance_formula r hr]",
        "rewrite_verified_distance_formula",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_817_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_geometry_817_json",
        "positive",
        "by\n  refine ⟨main_circle_radius / (Real.sqrt 2 + 1), ?_, ?_, ?_⟩\n  · positivity\n  · exact (solve_tangency_equation _).2 rfl\n  · exact rationalize_denominator",
        "construct_radius_from_solved_tangency_equation",
    ),
    *[
        proof(
            "robustpa_aime_2024_88", name, "positive",
            f"by\n  norm_num [{defs}]",
            "unfold_radius_constants_and_normalize",
        )
        for name, defs in [
            ("r_i_value", "major_radius, minor_radius"),
            ("r_o_value", "major_radius, minor_radius"),
            ("radius_difference", "major_radius, minor_radius"),
            ("fraction_equals_difference", "m_value, n_value"),
        ]
    ],
    proof(
        "robustpa_aime_2024_88",
        "robustpa_qwen3_8b_math_verify_aime_2024_robustpa_aime_2024_88",
        "positive",
        "by\n  refine ⟨6, 1, by norm_num, by norm_num, by norm_num, ?_, by norm_num⟩\n  norm_num [major_radius, minor_radius]",
        "explicit_coprime_fraction_witness",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_intermediate_algebra_662_json", name, "positive",
            f"by\n  simp only [{lhs}, {rhs}]\n  field_simp\n  ring",
            "unfold_rational_transform_and_clear_denominators",
        )
        for name, lhs, rhs in [
            ("first_term_transform", "first_term_y", "y_subst"),
            ("second_term_transform", "second_term_y", "y_subst"),
            ("third_term_transform", "third_term_y", "y_subst"),
            ("first_term_y_to_z", "first_term_y", "first_term_z, z_subst"),
            ("second_term_y_to_z", "second_term_y", "second_term_z, z_subst"),
            ("third_term_y_to_z", "third_term_y", "third_term_z, z_subst"),
        ]
    ],
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "z_eq_10_solution", "positive",
        "by\n  norm_num [transformed_lhs_z, first_term_z, second_term_z, third_term_z]",
        "evaluate_transformed_rational_expression",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "y_value_from_z", "positive",
        "by\n  norm_num [z_subst]",
        "evaluate_substitution",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "quadratic_solution_plus", "positive",
        "by\n  have hs : (Real.sqrt 19) ^ 2 = (19 : ℝ) := by norm_num\n  nlinarith",
        "sqrt_square_solves_quadratic",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "plus_not_excluded", "positive",
        "by\n  norm_num [excluded_values]\n  all_goals have hp : 0 < Real.sqrt 19 := by positivity\n  all_goals nlinarith",
        "positive_square_root_excludes_listed_values",
    ),
    proof(
        "robustpa_cmimc_2025_11",
        "robustpa_qwen3_8b_math_verify_cmimc_2025_robustpa_cmimc_2025_11",
        "positive",
        "by\n  norm_num [probability_value]",
        "normalize_defined_probability",
    ),
    proof(
        "robustpa_hmmt_feb_2025_15",
        "robustpa_qwen3_8b_math_verify_hmmt_feb_2025_robustpa_hmmt_feb_2025_15",
        "positive",
        "by\n  exact max_doors_count",
        "reuse_verified_maximum_door_count",
    ),
    proof(
        "robustpa_MATH_500_test_counting_and_probability_430_json", "bags_original_iff", "positive",
        "by\n  simp [alice_final, bob_final, initial_bag]",
        "unfold_multiset_exchange",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_817_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_geometry_817_json",
        "positive",
        "by\n  have hs : (Real.sqrt 2) ^ 2 = (2 : ℝ) := by norm_num\n  have hs1 : 1 < Real.sqrt 2 := by nlinarith [Real.sqrt_nonneg 2]\n  refine ⟨main_circle_radius / (Real.sqrt 2 + 1), by positivity, ?_, ?_⟩\n  · exact (solve_tangency_equation _).2 rfl\n  · exact rationalize_denominator",
        "construct_positive_radius_from_solved_equation",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_intermediate_algebra_662_json", name, "positive",
            f"by\n  simp only [{lhs}, {rhs}]\n  field_simp [h1, h2, h3]\n  ring",
            "clear_denominators_using_exclusion_hypotheses",
        )
        for name, lhs, rhs in [
            ("second_term_y_to_z", "second_term_y", "second_term_z, z_subst"),
            ("third_term_y_to_z", "third_term_y", "third_term_z, z_subst"),
        ]
    ],
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "plus_not_excluded", "positive",
        "by\n  simp only [excluded_values, Set.mem_insert_iff, Set.mem_singleton_iff, not_or]\n  have hs : (Real.sqrt 19) ^ 2 = (19 : ℝ) := by norm_num\n  constructor <;> try constructor <;> try constructor <;> try constructor <;> try constructor\n  all_goals intro heq\n  all_goals nlinarith [Real.sqrt_nonneg 19]",
        "sqrt_square_excludes_each_finite_value",
    ),
    proof(
        "robustpa_cmimc_2025_11",
        "robustpa_qwen3_8b_math_verify_cmimc_2025_robustpa_cmimc_2025_11",
        "positive",
        "by\n  rw [probability_value, favorable_arrangements_lemma, burnside_lemma_application]\n  norm_num",
        "rewrite_verified_counts_and_normalize_ratio",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_geometry_826_json", name, "positive",
            f"by\n  norm_num [{defs}]",
            "unfold_coordinates_and_normalize",
        )
        for name, defs in [
            ("square_side_length", "point_C, point_D, Prod.dist_eq, Real.dist_eq"),
            ("vector_BE_def", "vector_BE, point_B, point_E"),
            ("original_BE_length", "vector_BE"),
            ("H_on_line_AG", "point_H"),
            ("BH_perp_BE", "point_H, point_B, vector_BE"),
        ]
    ],
    proof(
        "robustpa_MATH_500_test_geometry_826_json", "scaling_factor_simplified", "positive",
        "by\n  simp only [scaling_factor]\n  have hs : (Real.sqrt 5) ^ 2 = (5 : ℝ) := by norm_num\n  have h45 : Real.sqrt 45 = 3 * Real.sqrt 5 := by\n    rw [show (45 : ℝ) = 9 * 5 by norm_num, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 9)]\n    norm_num\n  rw [h45]\n  field_simp\n  nlinarith [Real.sqrt_nonneg 5]",
        "simplify_square_root_scaling_factor",
    ),
    proof(
        "robustpa_aime_2025_15", "equiv_all_div_27", "negative",
        "by\n  intro h\n  have hbad := h 1 3 314 (by norm_num) (by norm_num [bound]) (by norm_num) (by norm_num [bound]) (by norm_num) (by norm_num [bound])\n  norm_num [divisor] at hbad",
        "bounded_counterexample_1_3_314",
    ),
    *[
        proof(
            "robustpa_hmmt_feb_2025_10", name, "positive",
            body,
            reason,
        )
        for name, body, reason in [
            ("sum_of_squares_eq", "by\n  rcases h with ⟨_, _, _, ha, hb, hc⟩\n  rw [ha, hb, hc]\n  ring", "substitute_cyclic_square_equations"),
            ("product_of_equations", "by\n  rcases h with ⟨_, _, _, ha, hb, hc⟩\n  rw [ha, hb, hc]", "substitute_cyclic_square_equations"),
            ("poly_factorization", "by\n  intro S\n  ring", "polynomial_normalization"),
            ("S_eq_2_root", "by\n  norm_num", "evaluate_polynomial_at_two"),
            ("S_eq_neg3_root", "by\n  norm_num", "evaluate_polynomial_at_negative_three"),
        ]
    ],
    proof(
        "robustpa_hmmt_feb_2025_16", "total_pairs_is_25200", "positive",
        "by\n  set_option maxRecDepth 10000 in\n    norm_num [total_rectangle_pairs_count, total_rectangles_count, grid_line_count, Nat.choose]",
        "bounded_unfolding_of_choose_recursion",
    ),
    proof(
        "robustpa_aime_2025_8", "solve_absolute_value", "positive",
        "by\n  ext k\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  constructor\n  · intro h\n    rw [abs_eq (by norm_num : (50 : ℝ) ≥ 0)] at h\n    rcases h with h | h <;> right_or_left <;> linarith\n  · intro h\n    rcases h with rfl | rfl <;> norm_num",
        "solve_absolute_value_by_two_linear_cases",
    ),
    proof(
        "robustpa_aime_2025_8", "sum_of_k_values", "positive",
        "by\n  norm_num",
        "normalize_rational_sum",
    ),
    proof(
        "robustpa_aime_2025_8", "gcd_65_4", "positive",
        "by\n  norm_num",
        "compute_gcd",
    ),
    *[
        proof(
            "robustpa_cmimc_2025_23", name, "positive",
            f"by\n  have hs : (Real.sqrt 3) ^ 2 = (3 : ℝ) := by norm_num\n  rw [show {expr} = {rhs} by\n    simp only [{defs}]\n    nlinarith]\n  norm_num",
            "reduce_coordinate_distance_radicand",
        )
        for name, expr, rhs, defs in [
            ("c_on_circle_A", "((point_C.1 - point_A.1)^2 + (point_C.2 - point_A.2)^2)", "(1 : ℝ)", "point_C, point_A"),
            ("c_on_circle_B", "((point_C.1 - point_B.1)^2 + (point_C.2 - point_B.2)^2)", "(1 : ℝ)", "point_C, point_B"),
            ("d_on_circle_A", "((point_D.1 - point_A.1)^2 + (point_D.2 - point_A.2)^2)", "(1 : ℝ)", "point_D, point_A"),
            ("d_on_circle_B", "((point_D.1 - point_B.1)^2 + (point_D.2 - point_B.2)^2)", "(1 : ℝ)", "point_D, point_B"),
            ("distance_CD", "((point_C.1 - point_D.1)^2 + (point_C.2 - point_D.2)^2)", "(3 : ℝ)", "point_C, point_D"),
            ("distance_EF", "((point_E.1 - point_F.1)^2 + (point_E.2 - point_F.2)^2)", "(9 : ℝ)", "point_E, point_F"),
        ]
    ],
    *[
        proof(
            "robustpa_cmimc_2025_23", name, "positive",
            f"by\n  have hs : (Real.sqrt 3) ^ 2 = (3 : ℝ) := by norm_num\n  simp only [pentagon_vertices, point_A, point_C, point_F, point_H, point_E, List.rotateLeft, List.zip, List.foldl]\n  {tactic}",
            "evaluate_shoelace_fold",
        )
        for name, tactic in [
            ("shoelace_sum1", "ring"),
            ("shoelace_sum2", "ring"),
        ]
    ],
    proof(
        "robustpa_cmimc_2025_35", "expand_square", "positive",
        "by\n  intro x f\n  ring",
        "integer_square_expansion",
    ),
    proof(
        "robustpa_MATH_500_test_precalculus_1056_json", "simplified_equation", "positive",
        "by\n  rw [← mul_left_cancel_iff_of_pos (show (0 : ℝ) < 6 by norm_num)]\n  rw [sum_scaled, combine_expansions]\n  ring_nf",
        "scale_both_equations_and_rewrite_expansions",
    ),
    proof(
        "robustpa_MATH_500_test_precalculus_1056_json", "S_is_sphere", "positive",
        "by\n  ext p\n  exact simplified_equation p.1 p.2.1 p.2.2",
        "set_extensionality_from_pointwise_equivalence",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "product_of_equations", "positive",
        "by\n  rcases h with ⟨_, _, _, ha, hb, hc⟩\n  calc\n    (a * b * c)^2 = a^2 * b^2 * c^2 := by ring\n    _ = (b + 6) * (c + 6) * (a + 6) := by rw [ha, hb, hc]",
        "expand_product_square_then_substitute",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "S_eq_2_root", "negative",
        "by\n  norm_num",
        "direct_arithmetic_refutes_claimed_root",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_826_json", "original_BH_length", "positive",
        "by\n  simp only [point_B, point_H, Prod.dist_eq, Real.dist_eq]\n  rw [show |(3 : ℝ) - 12 / 5| = 3 / 5 by norm_num, show |(0 : ℝ) - 6 / 5| = 6 / 5 by norm_num]\n  rw [show (3 / 5 : ℝ) ^ 2 + (6 / 5 : ℝ) ^ 2 = 9 / 5 by norm_num]\n  have hs : Real.sqrt (9 / 5) = 3 * Real.sqrt 5 / 5 := by\n    have h5 : (Real.sqrt 5)^2 = (5 : ℝ) := by norm_num\n    apply (Real.sq_eq_sq₀ (by positivity) (by positivity)).mp\n    nlinarith\n  exact hs",
        "compute_product_metric_distance",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_826_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_geometry_826_json",
        "positive",
        "by\n  rw [scaling_factor_simplified]\n  have hs : (Real.sqrt 5)^2 = (5 : ℝ) := by norm_num\n  nlinarith",
        "multiply_simplified_scaling_factors",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_817_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_geometry_817_json",
        "positive",
        "by\n  have hs : (Real.sqrt 2) ^ 2 = (2 : ℝ) := by norm_num\n  have hs1 : 1 < Real.sqrt 2 := by nlinarith [Real.sqrt_nonneg 2]\n  refine ⟨main_circle_radius * (Real.sqrt 2 - 1), ?_, ?_, rfl⟩\n  · simp only [main_circle_radius]\n    positivity\n  · simp only [main_circle_radius]\n    nlinarith",
        "construct_rationalized_positive_radius",
    ),
    proof(
        "robustpa_MATH_500_test_geometry_826_json", "original_BH_length", "negative",
        "by\n  intro h\n  simp only [point_B, point_H, Prod.dist_eq, Real.dist_eq] at h\n  norm_num at h\n  have hs : (Real.sqrt 5)^2 = (5 : ℝ) := by norm_num\n  nlinarith [Real.sqrt_nonneg 5]",
        "sup_metric_distance_refutes_euclidean_claim",
    ),
    proof(
        "robustpa_MATH_500_test_precalculus_1056_json", "simplified_equation", "positive",
        "by\n  constructor <;> intro h\n  · have hs := sum_scaled x y z\n    have hc := combine_expansions x y z\n    nlinarith\n  · have hs := sum_scaled x y z\n    have hc := combine_expansions x y z\n    nlinarith",
        "linear_arithmetic_using_scaled_identity",
    ),
    proof(
        "robustpa_MATH_500_test_precalculus_1056_json", "enclosed_volume_description", "negative",
        "by\n  intro h\n  have h0 := (h 0 0 0).mp (by norm_num)\n  have hm := interior_subset h0\n  norm_num at hm",
        "origin_is_not_on_sphere_surface",
    ),
    proof(
        "robustpa_cmimc_2025_23", "distance_CD", "positive",
        "by\n  have hs : (Real.sqrt 3) ^ 2 = (3 : ℝ) := by norm_num\n  rw [show ((point_C.1 - point_D.1)^2 + (point_C.2 - point_D.2)^2) = (3 : ℝ) by\n    simp only [point_C, point_D]\n    nlinarith]",
        "coordinate_radicand_equals_three",
    ),
    *[
        proof(
            "robustpa_cmimc_2025_23", name, "positive",
            f"by\n  have hs : (Real.sqrt 3) ^ 2 = (3 : ℝ) := by norm_num\n  norm_num [pentagon_vertices, point_A, point_C, point_F, point_H, point_E, List.rotateLeft]\n  ring",
            "normalize_finite_shoelace_list",
        )
        for name in ["shoelace_sum1", "shoelace_sum2"]
    ],
    proof(
        "robustpa_hmmt_feb_2025_10", "product_of_equations", "positive",
        "by\n  rcases h with ⟨_, _, _, ha, hb, hc⟩\n  calc\n    (a * b * c)^2 = a^2 * b^2 * c^2 := by ring\n    _ = (b + 6) * (c + 6) * (a + 6) := by rw [ha, hb, hc]",
        "expand_product_square_then_rewrite",
    ),
    proof(
        "robustpa_cmimc_2025_39",
        "robustpa_qwen3_8b_math_verify_cmimc_2025_robustpa_cmimc_2025_39",
        "positive",
        "by\n  norm_num [area_L1L2L3, point_A, point_B, point_C, vector_AB, vector_BC, vector_AC, vector_BX, vector_CP, vector_CM, point_X, point_Y, point_P, point_Q, point_M, point_N, point_L1, point_L2, point_L3, abs_of_nonneg, abs_of_neg]",
        "unfold_coordinates_and_shoelace_area",
    ),
    *[
        proof(
            "robustpa_aime_2025_14", name, "positive",
            f"by\n  norm_num [{defs}, Prod.dist_eq, Real.dist_eq]",
            "compute_sup_metric_axis_aligned_distance",
        )
        for name, defs in [
            ("pentagon_BC", "point_B, point_C"),
            ("dist_BX", "optimal_X, point_B"),
            ("dist_CX", "optimal_X, point_C"),
        ]
    ],
    proof(
        "robustpa_aime_2025_14", "line_EG_slope", "positive",
        "by\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  simp only [point_E, point_G]\n  ring",
        "coordinate_slope_calculation",
    ),
    *[
        proof(
            "robustpa_aime_2025_14", name, "negative",
            f"by\n  intro h\n  simp only [{defs}, Prod.dist_eq, Real.dist_eq] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  norm_num [max_eq_left, max_eq_right] at h ⊢\n  nlinarith",
            "sup_product_metric_refutes_euclidean_distance_claim",
        )
        for name, defs in [
            ("pentagon_AB", "point_A, point_B"),
            ("pentagon_CD", "point_C, point_D"),
            ("pentagon_DE", "point_D, point_E"),
            ("pentagon_EA", "point_E, point_A"),
            ("dist_AX", "optimal_X, point_A"),
            ("dist_DX", "optimal_X, point_D"),
            ("dist_EX", "optimal_X, point_E"),
        ]
    ],
    proof(
        "robustpa_MATH_500_test_prealgebra_378_json", "first_in_second", "negative",
        "by\n  intro h\n  have hp : ((0, 0) : ℝ × ℝ) ∈ first_triangle := by norm_num [first_triangle]\n  have := h hp\n  norm_num [second_triangle] at this",
        "origin_counterexample_to_subset_claim",
    ),
    proof(
        "robustpa_hmmt_feb_2025_25", "right_angle_at_X", "negative",
        "by\n  intro h\n  have hc : X_on_circle_diameter_AP ((0,0) : ℝ × ℝ) (2,2) (0,1) := by\n    norm_num [X_on_circle_diameter_AP, Prod.dist_eq, Real.dist_eq]\n  have := h (0,0) (2,2) (0,1) hc\n  norm_num at this",
        "sup_metric_circle_counterexample_to_thales_claim",
    ),
    proof(
        "robustpa_aime_2025_14", "pentagon_AB", "negative",
        "by\n  intro h\n  simp only [point_A, point_B, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  rw [abs_of_nonneg hp] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hm : max 7 (7 * Real.sqrt 3) < 14 := max_lt (by norm_num) (by nlinarith)\n  linarith",
        "sup_metric_distance_is_strictly_less_than_euclidean_claim",
    ),
    proof(
        "robustpa_aime_2025_14", "pentagon_CD", "negative",
        "by\n  intro h\n  simp only [point_C, point_D, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  rw [abs_of_nonneg (by positivity : 0 ≤ 36 * Real.sqrt 3 / 7)] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hm : max (156/7 : ℝ) (36 * Real.sqrt 3 / 7) < 24 := max_lt (by norm_num) (by nlinarith)\n  linarith",
        "sup_metric_distance_is_strictly_less_than_claim",
    ),
    proof(
        "robustpa_aime_2025_14", "pentagon_DE", "negative",
        "by\n  intro h\n  simp only [point_D, point_E, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have habs : |36 * Real.sqrt 3 / 7 - 88 * Real.sqrt 3 / 7| = 52 * Real.sqrt 3 / 7 := by\n    rw [abs_of_nonpos (by positivity)]\n    ring\n  rw [habs] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hm : max (13/7 : ℝ) (52 * Real.sqrt 3 / 7) < 13 := max_lt (by norm_num) (by nlinarith)\n  linarith",
        "sup_metric_distance_is_strictly_less_than_claim",
    ),
    proof(
        "robustpa_aime_2025_14", "pentagon_EA", "negative",
        "by\n  intro h\n  simp only [point_E, point_A, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have habs : |88 * Real.sqrt 3 / 7 - 7 * Real.sqrt 3| = 39 * Real.sqrt 3 / 7 := by\n    rw [abs_of_nonneg (by positivity)]\n    ring\n  rw [habs] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hm : max (169/7 : ℝ) (39 * Real.sqrt 3 / 7) < 26 := max_lt (by norm_num) (by nlinarith)\n  linarith",
        "sup_metric_distance_is_strictly_less_than_claim",
    ),
    proof(
        "robustpa_aime_2025_14", "dist_AX", "negative",
        "by\n  intro h\n  simp only [optimal_X, point_A, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  rw [abs_of_nonneg hp] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hm : max 21 (7 * Real.sqrt 3) < 14 * Real.sqrt 3 := max_lt (by nlinarith) (by nlinarith)\n  linarith",
        "sup_metric_distance_is_strictly_less_than_euclidean_claim",
    ),
    proof(
        "robustpa_aime_2025_14", "dist_DX", "negative",
        "by\n  intro h\n  simp only [optimal_X, point_D, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  rw [abs_of_nonneg (by positivity : 0 ≤ 36 * Real.sqrt 3 / 7)] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hm : max (9/7 : ℝ) (36 * Real.sqrt 3 / 7) < 9 := max_lt (by norm_num) (by nlinarith)\n  linarith",
        "sup_metric_distance_is_strictly_less_than_claim",
    ),
    proof(
        "robustpa_aime_2025_14", "dist_EX", "negative",
        "by\n  intro h\n  simp only [optimal_X, point_E, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  rw [abs_of_nonneg (by positivity : 0 ≤ 88 * Real.sqrt 3 / 7)] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hm : max (22/7 : ℝ) (88 * Real.sqrt 3 / 7) < 22 := max_lt (by norm_num) (by nlinarith)\n  linarith",
        "sup_metric_distance_is_strictly_less_than_claim",
    ),
    *[
        proof(
            "robustpa_aime_2025_14", name, "negative",
            f"by\n  intro h\n  simp only [{defs}, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  norm_num [abs_of_nonneg, abs_of_nonpos] at h\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  {bounds}\n  linarith",
            "normalize_sup_metric_and_bound_both_coordinates",
        )
        for name, defs, bounds in [
            ("pentagon_AB", "point_A, point_B", "have hm : max 7 (7 * Real.sqrt 3) < 14 := max_lt (by norm_num) (by nlinarith)"),
            ("pentagon_CD", "point_C, point_D", "have hm : max (156/7 : ℝ) (36 * Real.sqrt 3 / 7) < 24 := max_lt (by norm_num) (by nlinarith)"),
            ("pentagon_DE", "point_D, point_E", "have hm : max (13/7 : ℝ) (52 * Real.sqrt 3 / 7) < 13 := max_lt (by norm_num) (by nlinarith)"),
            ("pentagon_EA", "point_E, point_A", "have hm : max (169/7 : ℝ) (39 * Real.sqrt 3 / 7) < 26 := max_lt (by norm_num) (by nlinarith)"),
            ("dist_AX", "optimal_X, point_A", "have hm : max 21 (7 * Real.sqrt 3) < 14 * Real.sqrt 3 := max_lt (by nlinarith) (by nlinarith)"),
            ("dist_DX", "optimal_X, point_D", "have hm : max (9/7 : ℝ) (36 * Real.sqrt 3 / 7) < 9 := max_lt (by norm_num) (by nlinarith)"),
            ("dist_EX", "optimal_X, point_E", "have hm : max (22/7 : ℝ) (88 * Real.sqrt 3 / 7) < 22 := max_lt (by norm_num) (by nlinarith)"),
        ]
    ],
    *[
        proof(
            "robustpa_aime_2025_14", name, "negative",
            f"by\n  intro h\n  simp only [{defs}, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have habs : {abs_expr} = {abs_value} := by\n    rw [{abs_rule}]\n    ring\n  rw [habs] at h\n  norm_num at h\n  {bounds}\n  nlinarith",
            "explicit_absolute_value_then_sup_metric_bound",
        )
        for name, defs, abs_expr, abs_value, abs_rule, bounds in [
            ("pentagon_AB", "point_A, point_B", "|7 * Real.sqrt 3 - 0|", "7 * Real.sqrt 3", "abs_of_nonneg (by positivity)", "skip"),
            ("pentagon_CD", "point_C, point_D", "|0 - 36 * Real.sqrt 3 / 7|", "36 * Real.sqrt 3 / 7", "abs_of_nonpos (by positivity)", "have hm : max (156/7 : ℝ) (36 * Real.sqrt 3 / 7) < 24 := max_lt (by norm_num) (by nlinarith)"),
            ("pentagon_DE", "point_D, point_E", "|36 * Real.sqrt 3 / 7 - 88 * Real.sqrt 3 / 7|", "52 * Real.sqrt 3 / 7", "abs_of_nonpos (by nlinarith)", "have hm : max (13/7 : ℝ) (52 * Real.sqrt 3 / 7) < 13 := max_lt (by norm_num) (by nlinarith)"),
            ("pentagon_EA", "point_E, point_A", "|88 * Real.sqrt 3 / 7 - 7 * Real.sqrt 3|", "39 * Real.sqrt 3 / 7", "abs_of_nonneg (by nlinarith)", "have hm : max (169/7 : ℝ) (39 * Real.sqrt 3 / 7) < 26 := max_lt (by norm_num) (by nlinarith)"),
            ("dist_DX", "optimal_X, point_D", "|0 - 36 * Real.sqrt 3 / 7|", "36 * Real.sqrt 3 / 7", "abs_of_nonpos (by positivity)", "have hm : max (9/7 : ℝ) (36 * Real.sqrt 3 / 7) < 9 := max_lt (by norm_num) (by nlinarith)"),
            ("dist_EX", "optimal_X, point_E", "|0 - 88 * Real.sqrt 3 / 7|", "88 * Real.sqrt 3 / 7", "abs_of_nonpos (by positivity)", "have hm : max (22/7 : ℝ) (88 * Real.sqrt 3 / 7) < 22 := max_lt (by norm_num) (by nlinarith)"),
        ]
    ],
    *[
        proof(
            "robustpa_aime_2025_14", name, "negative",
            f"by\n  intro h\n  simp only [{defs}, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have habs : {abs_expr} = {abs_value} := by\n    rw [{abs_rule}]\n    ring\n  rw [habs] at h\n  norm_num at h\n  {bounds}\n  nlinarith",
            "explicit_absolute_value_then_nonlinear_bound",
        )
        for name, defs, abs_expr, abs_value, abs_rule, bounds in [
            ("pentagon_AB", "point_A, point_B", "|7 * Real.sqrt 3 - 0|", "7 * Real.sqrt 3", "abs_of_nonneg (by nlinarith)", "skip"),
            ("pentagon_CD", "point_C, point_D", "|0 - 36 * Real.sqrt 3 / 7|", "36 * Real.sqrt 3 / 7", "abs_of_nonpos (by nlinarith)", "have hm : max (156/7 : ℝ) (36 * Real.sqrt 3 / 7) < 24 := max_lt (by norm_num) (by nlinarith)"),
            ("dist_DX", "optimal_X, point_D", "|0 - 36 * Real.sqrt 3 / 7|", "36 * Real.sqrt 3 / 7", "abs_of_nonpos (by nlinarith)", "have hm : max (9/7 : ℝ) (36 * Real.sqrt 3 / 7) < 9 := max_lt (by norm_num) (by nlinarith)"),
            ("dist_EX", "optimal_X, point_E", "|0 - 88 * Real.sqrt 3 / 7|", "88 * Real.sqrt 3 / 7", "abs_of_nonpos (by nlinarith)", "have hm : max (22/7 : ℝ) (88 * Real.sqrt 3 / 7) < 22 := max_lt (by norm_num) (by nlinarith)"),
        ]
    ],
    proof(
        "robustpa_aime_2025_27", "alternating_pattern", "positive",
        "by\n  refine ⟨1, 26/5, by norm_num, by norm_num, by norm_num, ?_⟩\n  intro i hi\n  split <;> split <;> norm_num at * ⊢\n  omega",
        "explicit_alternating_pair_witness",
    ),
    proof(
        "robustpa_aime_2025_27", "law_of_cosines_relation", "positive",
        "by\n  intro a b ha hb hab\n  nlinarith [sq_nonneg (a - b)]",
        "am_gm_bound_from_fixed_product",
    ),
    proof(
        "robustpa_aime_2025_27",
        "robustpa_qwen3_8b_math_verify_aime_2025_robustpa_aime_2025_27",
        "positive",
        "by\n  norm_num\n  intro prime hp hsq\n  interval_cases prime <;> norm_num at hp hsq",
        "finite_arithmetic_certificate_for_output_tuple",
    ),
    proof(
        "robustpa_cmimc_2025_7", "sum_first_n_integers", "positive",
        "by\n  induction n with\n  | zero => norm_num\n  | succ n ih =>\n      rw [Finset.sum_range_succ, ih]\n      omega",
        "induction_on_arithmetic_series",
    ),
    proof(
        "robustpa_cmimc_2025_7", "nu_3_at_multiples_of_9", "negative",
        "by\n  intro h\n  have h1 := h 1 (by norm_num)\n  norm_num [nu_3, a, last_digit, padicValNat] at h1",
        "evaluate_first_multiple_of_nine",
    ),
    proof(
        "robustpa_aime_2025_27", "alternating_pattern", "positive",
        "by\n  refine ⟨1, 26/5, by norm_num, by norm_num, by norm_num, ?_⟩\n  intro i hi\n  have hpar : i % 2 = 0 ∨ i % 2 = 1 := Nat.mod_two_eq_zero_or_one i\n  rcases hpar with he | he\n  · simp [he, Nat.add_mod]\n  · simp [he, Nat.add_mod]",
        "explicit_pair_and_parity_cases",
    ),
    proof(
        "robustpa_hmmt_feb_2025_4", "sum_rewrite", "positive",
        "by\n  have hn : ((j : ℝ) + 0.5) ≠ 0 := by\n    intro h\n    have : (2 * j + 1 : ℤ) = 0 := by exact_mod_cast (by linarith : (2 * (j : ℝ) + 1) = 0)\n    omega\n  have hn2 : (((2 * j + 1 : ℤ) : ℝ)) ≠ 0 := by\n    exact_mod_cast (show (2 * j + 1 : ℤ) ≠ 0 by omega)\n  field_simp\n  ring",
        "clear_nonzero_half_integer_denominators",
    ),
    proof(
        "robustpa_hmmt_feb_2025_4", "count_positive_odd_divisors", "positive",
        "by\n  norm_num [positive_odd_divisors_le_1999]\n  decide",
        "finite_filtered_interval_computation",
    ),
    proof(
        "robustpa_aime_2025_8",
        "robustpa_qwen3_8b_math_verify_aime_2025_robustpa_aime_2025_8",
        "positive",
        "by\n  norm_num",
        "normalize_final_tuple",
    ),
    proof(
        "robustpa_hmmt_feb_2025_4", "sum_rewrite", "positive",
        "by\n  have hn : ((j : ℝ) + 0.5) ≠ 0 := by\n    intro h\n    have : (2 * j + 1 : ℤ) = 0 := by exact_mod_cast (by linarith : (2 * (j : ℝ) + 1) = 0)\n    omega\n  have hn2 : (((2 * j + 1 : ℤ) : ℝ)) ≠ 0 := by\n    exact_mod_cast (show (2 * j + 1 : ℤ) ≠ 0 by omega)\n  field_simp\n  push_cast\n  ring",
        "clear_denominators_and_normalize_integer_casts",
    ),
    proof(
        "robustpa_cmimc_2025_38", "segment_length_value", "positive",
        "by\n  norm_num [segment_length]",
        "evaluate_segment_length",
    ),
    proof(
        "robustpa_aime_2025_11", "c_squarefree", "positive",
        "by\n  norm_num [show (273 : ℕ) = 3 * 7 * 13 by norm_num]",
        "prime_factorization_squarefree_check",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "original_to_y_transform", "positive",
        "by\n  simp only [original_lhs, transformed_lhs_y]\n  rw [first_term_transform x h, second_term_transform x h, third_term_transform x h]",
        "assemble_three_verified_term_transforms",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "second_term_y_to_z", "positive",
        "by\n  simp only [second_term_y, second_term_z, z_subst]\n  field_simp [sub_ne_zero.mpr h2]\n  ring",
        "clear_denominator_using_y_not_24",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "third_term_y_to_z", "positive",
        "by\n  simp only [third_term_y, third_term_z, z_subst]\n  field_simp [sub_ne_zero.mpr h3]\n  ring",
        "clear_denominator_using_y_not_48",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "y_value_plus", "positive",
        "by\n  simp only [y_subst]\n  nlinarith [quadratic_solution_plus]",
        "reuse_quadratic_identity",
    ),
    *[
        proof(
            "robustpa_MATH_500_test_intermediate_algebra_662_json", name, "positive",
            f"by\n  simp only [{defs}]\n  congr 1 <;> ring",
            "numerator_and_denominator_polynomial_identity",
        )
        for name, defs in [
            ("second_term_y_to_z", "second_term_y, second_term_z, z_subst"),
            ("third_term_y_to_z", "third_term_y, third_term_z, z_subst"),
        ]
    ],
    proof(
        "robustpa_hmmt_feb_2025_10", "S2_minus_2P_eq", "positive",
        "by\n  rw [square_sum_identity]\n  exact sum_of_squares_eq a b c h",
        "rewrite_square_sum_identity",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "P_formula", "positive",
        "by\n  have hs := S2_minus_2P_eq a b c h\n  linear_combination -(1/2) * hs",
        "solve_linear_equation_for_symmetric_product",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q2_eq", "positive",
        "by\n  rw [product_of_equations a b c h, expand_product]",
        "rewrite_product_expansion",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q2_sub_P", "positive",
        "by\n  rw [Q2_eq a b c h, P_formula a b c h]",
        "substitute_symmetric_product_formula",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q2_simplified", "positive",
        "by\n  have hq := Q2_sub_P a b c h\n  linear_combination hq",
        "normalize_symmetric_polynomial_equation",
    ),
    *[
        proof(
            "robustpa_cmimc_2025_23", name, "positive",
            f"by\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  rw [show {expr} = ({rhs} : ℝ) by\n    simp only [{defs}]\n    nlinarith]",
            "coordinate_radicand_reduction",
        )
        for name, expr, rhs, defs in [
            ("e_on_circle_C", "((point_E.1 - point_C.1)^2 + (point_E.2 - point_C.2)^2)", "3", "point_E, point_C"),
            ("e_on_circle_D", "((point_E.1 - point_D.1)^2 + (point_E.2 - point_D.2)^2)", "3", "point_E, point_D"),
            ("f_on_circle_C", "((point_F.1 - point_C.1)^2 + (point_F.2 - point_C.2)^2)", "3", "point_F, point_C"),
            ("f_on_circle_D", "((point_F.1 - point_D.1)^2 + (point_F.2 - point_D.2)^2)", "3", "point_F, point_D"),
            ("g_on_circle_E", "((point_G.1 - point_E.1)^2 + (point_G.2 - point_E.2)^2)", "9", "point_G, point_E"),
            ("g_on_circle_F", "((point_G.1 - point_F.1)^2 + (point_G.2 - point_F.2)^2)", "9", "point_G, point_F"),
            ("h_on_circle_E", "((point_H.1 - point_E.1)^2 + (point_H.2 - point_E.2)^2)", "9", "point_H, point_E"),
            ("h_on_circle_F", "((point_H.1 - point_F.1)^2 + (point_H.2 - point_F.2)^2)", "9", "point_H, point_F"),
        ]
    ],
    proof(
        "robustpa_cmimc_2025_23",
        "robustpa_qwen3_8b_math_verify_cmimc_2025_robustpa_cmimc_2025_23",
        "positive",
        "by\n  simp only [shoelace_formula]\n  rw [shoelace_sum1, shoelace_sum2]\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  rw [abs_of_nonneg (by nlinarith)]\n  ring",
        "assemble_verified_shoelace_sums",
    ),
    *[
        proof(
            "robustpa_cmimc_2025_23", name, "positive",
            f"by\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  rw [show {expr} = (9 : ℝ) by\n    simp only [{defs}]\n    nlinarith]\n  norm_num",
            "coordinate_radicand_nine_then_sqrt",
        )
        for name, expr, defs in [
            ("g_on_circle_E", "((point_G.1 - point_E.1)^2 + (point_G.2 - point_E.2)^2)", "point_G, point_E"),
            ("g_on_circle_F", "((point_G.1 - point_F.1)^2 + (point_G.2 - point_F.2)^2)", "point_G, point_F"),
            ("h_on_circle_E", "((point_H.1 - point_E.1)^2 + (point_H.2 - point_E.2)^2)", "point_H, point_E"),
            ("h_on_circle_F", "((point_H.1 - point_F.1)^2 + (point_H.2 - point_F.2)^2)", "point_H, point_F"),
        ]
    ],
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "y_to_z_transform", "positive",
        "by\n  simp only [transformed_lhs_y, transformed_lhs_z]\n  rw [first_term_y_to_z y h1 h2 h3, second_term_y_to_z y h1 h2 h3, third_term_y_to_z y h1 h2 h3]",
        "assemble_verified_y_to_z_terms",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json", "y_eq_18_solution", "positive",
        "by\n  rw [y_to_z_transform 18 (by norm_num) (by norm_num) (by norm_num), y_value_from_z, z_eq_10_solution]",
        "rewrite_verified_substitution_chain",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_662_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_intermediate_algebra_662_json",
        "positive",
        "by\n  rw [original_to_y_transform _ plus_not_excluded, y_value_plus, y_eq_18_solution]",
        "rewrite_full_transformation_chain",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "S2_minus_2P_eq", "positive",
        "by\n  calc\n    (a + b + c)^2 - 2*(a*b + b*c + c*a) = a^2 + b^2 + c^2 := by ring\n    _ = (a + b + c) + 18 := sum_of_squares_eq a b c h",
        "polynomial_identity_then_square_sum",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q2_eq", "positive",
        "by\n  rw [product_of_equations a b c h, expand_product]\n  ring",
        "expand_and_commute_symmetric_terms",
    ),
    proof(
        "robustpa_aime_2025_15", "cube_factorization", "positive",
        "by\n  subst x\n  rw [mul_pow]\n  congr 1\n  simp [pow_mul, mul_comm]",
        "power_of_product_and_exponent_multiplication",
    ),
    proof(
        "robustpa_aime_2025_27", "perimeter_equation", "positive",
        "by\n  dsimp only\n  have harg : a ^ 2 + b ^ 2 - 48/5 = (a + b)^2 - 20 := by nlinarith\n  rw [harg]\n  ring_nf",
        "fixed_product_identifies_both_square_root_arguments",
    ),
    proof(
        "robustpa_aime_2025_27", "solve_for_S", "positive",
        "by\n  dsimp only\n  have h95 : (Real.sqrt 95)^2 = (95 : ℝ) := by norm_num\n  have hs95 : 0 ≤ Real.sqrt 95 := Real.sqrt_nonneg 95\n  constructor\n  · nlinarith\n  · have hrad : 0 ≤ ((9 * Real.sqrt 95 - 25) / 14)^2 - 20 := by nlinarith\n    have hsquare := Real.sq_sqrt hrad\n    have hsnonneg := Real.sqrt_nonneg (((9 * Real.sqrt 95 - 25) / 14)^2 - 20)\n    nlinarith",
        "verify_radical_solution_by_squaring_with_signs",
    ),
    proof(
        "robustpa_cmimc_2025_23",
        "robustpa_qwen3_8b_math_verify_cmimc_2025_robustpa_cmimc_2025_23",
        "positive",
        "by\n  simp only [shoelace_formula]\n  rw [shoelace_sum1, shoelace_sum2]\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  rw [abs_of_nonneg (by nlinarith)]\n  ring",
        "assemble_shoelace_formula_after_all_coordinate_checks",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "P_formula", "positive",
        "by\n  have hs := S2_minus_2P_eq a b c h\n  linear_combination -(1/2) * hs",
        "solve_symmetric_linear_equation",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q2_sub_P", "positive",
        "by\n  rw [Q2_eq a b c h, P_formula a b c h]",
        "substitute_pairwise_product",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q2_simplified", "positive",
        "by\n  have hq := Q2_sub_P a b c h\n  linear_combination hq",
        "normalize_after_pairwise_product_substitution",
    ),
    proof(
        "robustpa_aime_2025_27", "solve_for_S", "negative",
        "by\n  intro h\n  rcases h with ⟨_, heq⟩\n  have h95 : (Real.sqrt 95)^2 = (95 : ℝ) := by norm_num\n  have hs95 : 0 ≤ Real.sqrt 95 := Real.sqrt_nonneg 95\n  have hgt : 9 < Real.sqrt 95 := by nlinarith\n  have hroot := Real.sqrt_nonneg (((9 * Real.sqrt 95 - 25) / 14)^2 - 20)\n  nlinarith",
        "extraneous_squared_root_makes_left_side_exceed_twenty",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_coeff_2_mod_3", "positive",
        "by\n  norm_num [trinomial_coeff, Polynomial.coeff_add, Polynomial.coeff_pow]",
        "expand_small_polynomial_power",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_coeff_0_mod_3", "positive",
        "by\n  constructor\n  · norm_num [trinomial_coeff]\n  · intro c hc\n    norm_num [trinomial_coeff, Polynomial.coeff_one]\n    omega",
        "coefficients_of_constant_polynomial",
    ),
    proof(
        "robustpa_cmimc_2025_37", "valid_c_for_digit_2", "positive",
        "by\n  intro c hc\n  interval_cases c <;> norm_num [trinomial_coeff, Polynomial.coeff_add, Polynomial.coeff_pow]",
        "enumerate_three_digit_choices",
    ),
    proof(
        "robustpa_cmimc_2025_37", "valid_c_for_digit_0", "positive",
        "by\n  intro c hc\n  interval_cases c <;> norm_num [trinomial_coeff]",
        "enumerate_three_digit_choices",
    ),
    proof(
        "robustpa_cmimc_2025_37", "count_valid_choices_per_digit", "positive",
        "by\n  norm_num [Finset.filter_eq, trinomial_coeff, Polynomial.coeff_add, Polynomial.coeff_pow]",
        "evaluate_filtered_three_element_ranges",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_coeff_2_mod_3", "positive",
        "by\n  norm_num [trinomial_coeff, pow_two, Polynomial.coeff_mul]",
        "expand_degree_two_polynomial_multiplication",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_coeff_0_mod_3", "positive",
        "by\n  simp [trinomial_coeff]\n  intro c hc hzero\n  omega",
        "coefficients_of_polynomial_one",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_coeff_2_mod_3", "positive",
        "by\n  have hp : ((1 + Polynomial.X + Polynomial.X^2 : Polynomial ℕ)^2) =\n      1 + 2 * Polynomial.X + 3 * Polynomial.X^2 + 2 * Polynomial.X^3 + Polynomial.X^4 := by ring\n  simp only [trinomial_coeff, hp]\n  norm_num",
        "polynomial_ring_expansion_then_coefficients",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_coeff_0_mod_3", "positive",
        "by\n  constructor\n  · norm_num [trinomial_coeff]\n  · intro c hc\n    simp [trinomial_coeff, Polynomial.coeff_one, Nat.ne_of_gt hc]",
        "constant_polynomial_coefficients",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_coeff_2_mod_3", "positive",
        "by\n  have hp : ((1 + Polynomial.X + Polynomial.X^2 : Polynomial ℕ)^2) =\n      1 + 2 * Polynomial.X + 3 * Polynomial.X^2 + 2 * Polynomial.X^3 + Polynomial.X^4 := by ring\n  simp [trinomial_coeff, hp, Polynomial.coeff_one, Polynomial.coeff_X, Polynomial.coeff_X_pow]",
        "expanded_polynomial_coefficient_simplification",
    ),
    proof(
        "robustpa_cmimc_2025_38", "p1_roots", "positive",
        "by\n  ext x\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  have hid : p1 x = (x - (-1 + 2 * Complex.I)) * (x - (-1 - 2 * Complex.I)) := by\n    simp [p1]\n    ring_nf\n  rw [hid, mul_eq_zero, sub_eq_zero, sub_eq_zero]",
        "factor_quadratic_over_complex_numbers",
    ),
    proof(
        "robustpa_cmimc_2025_25", "distance_P1_P2", "negative",
        "by\n  intro h\n  simp only [P1, P2, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  norm_num [abs_of_nonneg, abs_of_nonpos] at h\n  split at h <;> nlinarith",
        "sup_metric_refutes_euclidean_distance_value",
    ),
    proof(
        "robustpa_cmimc_2025_25", "P1_on_E_AB", "negative",
        "by\n  intro h\n  simp only [ellipse_E_AB, P1, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  norm_num [abs_of_nonneg, abs_of_nonpos] at h\n  split at h <;> split at h <;> nlinarith",
        "sup_metric_refutes_ellipse_membership",
    ),
    proof(
        "robustpa_cmimc_2025_38", "p1_roots", "positive",
        "by\n  ext x\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  have hid : p1 x = (x - (-1 + 2 * Complex.I)) * (x - (-1 - 2 * Complex.I)) := by\n    simp [p1, Complex.I_mul_I]\n    ring\n  rw [hid, mul_eq_zero, sub_eq_zero, sub_eq_zero]",
        "factor_quadratic_using_i_squared",
    ),
    proof(
        "robustpa_cmimc_2025_25", "distance_P1_P2", "negative",
        "by\n  intro h\n  simp only [P1, P2, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hgt : 1 < Real.sqrt 3 := by nlinarith\n  have hx : |(17 - 12 * Real.sqrt 3) / 26 - (17 + 12 * Real.sqrt 3) / 26| = 12 * Real.sqrt 3 / 13 := by\n    rw [abs_of_nonpos (by nlinarith)]\n    ring\n  have hy : |(12 + 3 * Real.sqrt 3) / 26 - (3 * Real.sqrt 3 - 12) / 26| = 12 / 13 := by\n    rw [abs_of_nonneg (by nlinarith)]\n    ring\n  rw [hx, hy, max_eq_left (by nlinarith)] at h\n  nlinarith",
        "normalize_sup_distance_and_contradict_sqrt_three",
    ),
    proof(
        "robustpa_cmimc_2025_25", "P1_on_E_AB", "negative",
        "by\n  intro h\n  simp only [ellipse_E_AB, P1, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hlow : 17/12 < Real.sqrt 3 := by nlinarith\n  have hupp : Real.sqrt 3 < 29/9 := by nlinarith\n  have hx0 : |(17 - 12 * Real.sqrt 3) / 26| = (12 * Real.sqrt 3 - 17) / 26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  have hy : |(12 + 3 * Real.sqrt 3) / 26| = (12 + 3 * Real.sqrt 3) / 26 := by rw [abs_of_nonneg (by nlinarith)]\n  have hx1 : |(17 - 12 * Real.sqrt 3) / 26 - 1| = (9 + 12 * Real.sqrt 3) / 26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  rw [hx0, hy, hx1, max_eq_right (by nlinarith), max_eq_left (by nlinarith)] at h\n  nlinarith",
        "evaluate_both_sup_distances_and_refute_sum_two",
    ),
    proof(
        "robustpa_cmimc_2025_38", "p1_roots", "positive",
        "by\n  ext x\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  have hi : Complex.I ^ 2 = (-1 : ℂ) := by norm_num\n  have hid : p1 x = (x - (-1 + 2 * Complex.I)) * (x - (-1 - 2 * Complex.I)) := by\n    simp only [p1]\n    ring_nf\n    rw [hi]\n    norm_num\n  rw [hid, mul_eq_zero, sub_eq_zero, sub_eq_zero]",
        "factor_quadratic_with_explicit_i_square",
    ),
    proof(
        "robustpa_cmimc_2025_25", "P1_on_E_AB", "negative",
        "by\n  intro h\n  simp only [ellipse_E_AB, P1, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  simp only [sub_zero] at h\n  have hp : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n  have hs : (Real.sqrt 3)^2 = (3 : ℝ) := by norm_num\n  have hlow : 17/12 < Real.sqrt 3 := by nlinarith\n  have hupp : Real.sqrt 3 < 29/9 := by nlinarith\n  have hx0 : |(17 - 12 * Real.sqrt 3) / 26| = (12 * Real.sqrt 3 - 17) / 26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  have hy : |(12 + 3 * Real.sqrt 3) / 26| = (12 + 3 * Real.sqrt 3) / 26 := by rw [abs_of_nonneg (by nlinarith)]\n  have hx1 : |(17 - 12 * Real.sqrt 3) / 26 - 1| = (9 + 12 * Real.sqrt 3) / 26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  rw [hx0, hy, hx1, max_eq_right (by nlinarith), max_eq_left (by nlinarith)] at h\n  nlinarith",
        "normalize_zero_subtractions_and_evaluate_sup_distances",
    ),
    proof(
        "robustpa_cmimc_2025_38", "p1_roots", "positive",
        "by\n  ext x\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  have hi : Complex.I ^ 2 = (-1 : ℂ) := by norm_num\n  have hid : p1 x = (x - (-1 + 2 * Complex.I)) * (x - (-1 - 2 * Complex.I)) := by\n    simp only [p1]\n    ring_nf\n    rw [hi]\n    ring\n  rw [hid, mul_eq_zero, sub_eq_zero, sub_eq_zero]",
        "factor_quadratic_and_normalize_constants",
    ),
    proof(
        "robustpa_cmimc_2025_37", "valid_c_for_digit_2", "positive",
        "by\n  rcases trinomial_coeff_2_mod_3 with ⟨h0, h1, h2⟩\n  intro c hc\n  interval_cases c <;> simp_all",
        "three_cases_from_verified_coefficients",
    ),
    proof(
        "robustpa_cmimc_2025_37", "valid_c_for_digit_0", "positive",
        "by\n  rcases trinomial_coeff_0_mod_3 with ⟨h0, hpos⟩\n  intro c hc\n  interval_cases c <;> simp_all",
        "three_cases_from_constant_coefficients",
    ),
    proof(
        "robustpa_cmimc_2025_37", "count_valid_choices_per_digit", "positive",
        "by\n  constructor <;> norm_num [Finset.filter_eq, valid_c_for_digit_2, valid_c_for_digit_0]",
        "filter_three_digit_choices",
    ),
    proof(
        "robustpa_aime_2025_14", "optimal_X_on_EG", "positive",
        "by\n  simp only [optimal_X, point_E]\n  ring",
        "coordinate_line_equation",
    ),
    proof(
        "robustpa_aime_2025_11", "c_squarefree", "positive",
        "by\n  rw [Nat.squarefree_iff_prime_squarefree]\n  intro p hp hd\n  have hp2 : p ^ 2 ≤ 273 := Nat.le_of_dvd (by norm_num) hd\n  have hp16 : p ≤ 16 := by nlinarith\n  interval_cases p <;> norm_num at hp hd",
        "bound_and_enumerate_possible_prime_square_divisors",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  change ({p : GridPos | p ≠ start_pos ∧ p ≠ end_pos} : Set GridPos).ncard = 7\n  convert Set.ncard_compl ({start_pos, end_pos} : Set GridPos) using 1 <;> norm_num [GridPos, start_pos, end_pos, Set.ncard_pair, Set.ncard_univ]\n  ext p\n  simp [start_pos, end_pos]",
        "complement_of_two_points_in_nine_element_grid",
    ),
    proof(
        "robustpa_cmimc_2025_38", "p1_roots", "positive",
        "by\n  ext x\n  simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]\n  have hi : Complex.I ^ 2 = (-1 : ℂ) := by norm_num\n  have hid : p1 x = (x - (-1 + 2 * Complex.I)) * (x - (-1 - 2 * Complex.I)) := by\n    simp only [p1]\n    ring_nf\n    linear_combination -4 * hi\n  rw [hid, mul_eq_zero, sub_eq_zero, sub_eq_zero]",
        "factor_quadratic_by_linear_combination_with_i_square",
    ),
    proof(
        "robustpa_aime_2025_11", "c_squarefree", "positive",
        "by\n  rw [Nat.squarefree_iff_prime_squarefree]\n  intro p hp hd\n  have hp2 : p * p ≤ 273 := Nat.le_of_dvd (by norm_num) hd\n  have hp16 : p ≤ 16 := by nlinarith\n  interval_cases p <;> norm_num at hp ⊢",
        "enumerate_prime_candidates_up_to_sixteen",
    ),
    proof(
        "robustpa_cmimc_2025_37", "count_valid_choices_per_digit", "positive",
        "by\n  have htwo : (Finset.filter (fun c => trinomial_coeff 2 c % 3 = 1) (Finset.range 3)) = {0} := by\n    ext c\n    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]\n    constructor\n    · rintro ⟨hc, hv⟩\n      exact (valid_c_for_digit_2 c hc).mp hv\n    · intro hc\n      subst c\n      norm_num [trinomial_coeff]\n  have hzero : (Finset.filter (fun c => trinomial_coeff 0 c % 3 = 1) (Finset.range 3)) = {0} := by\n    ext c\n    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]\n    constructor\n    · rintro ⟨hc, hv⟩\n      exact (valid_c_for_digit_0 c hc).mp hv\n    · intro hc\n      subst c\n      norm_num [trinomial_coeff]\n  rw [htwo, hzero]\n  norm_num",
        "identify_each_filtered_range_with_singleton_zero",
    ),
    proof(
        "robustpa_cmimc_2025_37", "count_valid_choices_per_digit", "positive",
        "by\n  rcases trinomial_coeff_2_mod_3 with ⟨h20, h21, h22⟩\n  rcases trinomial_coeff_0_mod_3 with ⟨h00, h0pos⟩\n  have htwo : (Finset.filter (fun c => trinomial_coeff 2 c % 3 = 1) (Finset.range 3)) = {0} := by\n    ext c\n    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]\n    constructor\n    · rintro ⟨hc, hv⟩\n      exact (valid_c_for_digit_2 c hc).mp hv\n    · intro hc\n      subst c\n      exact ⟨by norm_num, h20⟩\n  have hzero : (Finset.filter (fun c => trinomial_coeff 0 c % 3 = 1) (Finset.range 3)) = {0} := by\n    ext c\n    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]\n    constructor\n    · rintro ⟨hc, hv⟩\n      exact (valid_c_for_digit_0 c hc).mp hv\n    · intro hc\n      subst c\n      exact ⟨by norm_num, h00⟩\n  rw [htwo, hzero]\n  norm_num",
        "singleton_filters_using_verified_zero_coefficients",
    ),
    proof(
        "robustpa_aime_2025_11", "c_squarefree", "positive",
        "by\n  rw [Nat.squarefree_iff_prime_squarefree]\n  intro p hp hd\n  have hp2 : p * p ≤ 273 := Nat.le_of_dvd (by norm_num) hd\n  have hp16 : p ≤ 16 := by nlinarith\n  interval_cases p\n  all_goals norm_num at hp\n  all_goals norm_num at hd",
        "prime_square_divisor_enumeration",
    ),
    proof(
        "robustpa_MATH_500_test_prealgebra_1139_json", "all_values_distinct", "positive",
        "by\n  rw [show claimed_values = ↑({121, 144, 126, 160, 150, 180, 71, 101, 51, 35, 47, 84} : Finset ℕ) by\n    ext x\n    simp [claimed_values]]\n  rw [Set.ncard_coe_finset]\n  norm_num",
        "represent_literal_set_as_finset_and_compute_cardinality",
    ),
    proof(
        "robustpa_MATH_500_test_prealgebra_1139_json", "obtainable_equals_claimed", "positive",
        "by\n  ext x\n  simp only [obtainable_values, Set.mem_range, claimed_values, Set.mem_insert_iff, Set.mem_singleton_iff]\n  constructor\n  · rintro ⟨p, rfl⟩\n    fin_cases p <;> norm_num [evaluate_with_paren]\n  · intro h\n    rcases h with h|h|h|h|h|h|h|h|h|h|h|h <;> subst x\n    · exact ⟨⟨0, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨1, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨2, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨3, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨4, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨5, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨6, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨7, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨8, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨9, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨10, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨11, by norm_num⟩, by norm_num [evaluate_with_paren]⟩",
        "enumerate_all_twelve_finite_parenthesization_indices",
    ),
    proof(
        "robustpa_MATH_500_test_prealgebra_1139_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_prealgebra_1139_json",
        "positive",
        "by\n  rw [obtainable_equals_claimed, all_values_distinct]",
        "rewrite_set_equality_and_verified_cardinality",
    ),
    proof(
        "robustpa_hmmt_feb_2025_4", "count_positive_odd_divisors", "positive",
        "by\n  set_option maxRecDepth 100000 in\n    norm_num [positive_odd_divisors_le_1999]",
        "bounded_finite_interval_computation",
    ),
    proof(
        "robustpa_MATH_500_test_prealgebra_1139_json", "obtainable_equals_claimed", "positive",
        "by\n  ext x\n  simp only [obtainable_values, Set.mem_range, claimed_values, Set.mem_insert_iff, Set.mem_singleton_iff]\n  constructor\n  · rintro ⟨p, rfl⟩\n    change Fin 12 at p\n    fin_cases p <;> norm_num [evaluate_with_paren]\n  · intro h\n    rcases h with h|h|h|h|h|h|h|h|h|h|h|h <;> subst x\n    · exact ⟨⟨0, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨1, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨2, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨3, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨4, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨5, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨6, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨7, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨8, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨9, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨10, by norm_num⟩, by norm_num [evaluate_with_paren]⟩\n    · exact ⟨⟨11, by norm_num⟩, by norm_num [evaluate_with_paren]⟩",
        "enumerate_fin_twelve_after_unfolding_defined_index_type",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  classical\n  letI : Fintype GridPos := inferInstance\n  have heq : removable_positions = ({start_pos, end_pos} : Set GridPos)ᶜ := by\n    ext p\n    simp [removable_positions, start_pos, end_pos]\n  rw [heq, Set.ncard_compl]\n  norm_num [GridPos, start_pos, end_pos, Set.ncard_pair, Set.ncard_univ]",
        "count_complement_of_two_distinct_points_in_fin_nine",
    ),
    proof(
        "robustpa_hmmt_feb_2025_4", "count_positive_odd_divisors", "negative",
        "by\n  set_option maxRecDepth 100000 in\n    decide",
        "kernel_decision_of_closed_finite_cardinality_counterclaim",
    ),
    proof(
        "robustpa_cmimc_2025_7", "sum_first_n_integers", "positive",
        "by\n  intro n\n  calc\n    (Finset.sum (Finset.range n) fun i => i + 1) = (Finset.sum (Finset.range n) fun i => i) + n := by\n      rw [Finset.sum_add_distrib]\n      simp\n    _ = Finset.sum (Finset.range (n + 1)) (fun i => i) := by\n      rw [Finset.sum_range_succ]\n    _ = n * (n + 1) / 2 := by\n      simpa [Nat.mul_comm] using Finset.sum_range_id (n := n + 1)",
        "shift_sum_to_standard_sum_range_identity",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_960_json", "triangle_ineq_equiv", "positive",
        "by\n  intro a b c ha hb hc\n  simp [is_triangle_bool, Bool.and_eq_true]\n  omega",
        "unfold_boolean_triangle_test_and_split_max_cases_arithmetically",
    ),
    proof(
        "robustpa_MATH_500_test_counting_and_probability_430_json", "bags_original_iff", "positive",
        "by\n  intro X Y hX hY\n  simp only [colors, Finset.mem_insert, Finset.mem_singleton] at hX hY\n  rcases hX with hX|hX|hX|hX|hX <;> rcases hY with hY|hY|hY|hY|hY <;>\n    subst X <;> subst Y <;> simp only [alice_final, bob_final, initial_bag, colors] <;> decide",
        "enumerate_the_five_by_five_color_pairs_and_compute_multisets",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  classical\n  rw [show removable_positions = (↑({(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)} : Finset (Fin 3 × Fin 3)) : Set (Fin 3 × Fin 3)) by\n    ext p\n    change Fin 3 × Fin 3 at p\n    fin_cases p.1 <;> fin_cases p.2 <;> simp [removable_positions, start_pos, end_pos]]\n  rw [Set.ncard_coe_finset]\n  norm_num",
        "enumerate_the_seven_nonendpoint_grid_positions",
    ),
    proof(
        "robustpa_hmmt_feb_2025_29", "right_angle_circumradius_AXD_relation", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (208 : ℝ)) ^ 2 = 208 := Real.sq_sqrt (by norm_num)\n  have hp : 0 < Real.sqrt (208 : ℝ) := Real.sqrt_pos.2 (by norm_num)\n  have hc := h 26 8 12 (by norm_num [point_X_inside]) (by norm_num [right_angle_condition]) (by\n    norm_num [circumradius_AXD_eq]\n    nlinarith)\n  norm_num at hc\n  field_simp at hc\n  nlinarith",
        "counterexample_rectangle_width_26_point_8_12_exposes_missing_factor_two",
    ),
    proof(
        "robustpa_hmmt_feb_2025_29", "right_angle_circumradius_BXC_relation", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (180 : ℝ)) ^ 2 = 180 := Real.sq_sqrt (by norm_num)\n  have hp : 0 < Real.sqrt (180 : ℝ) := Real.sqrt_pos.2 (by norm_num)\n  have hc := h 30 24 12 (by norm_num [point_X_inside]) (by norm_num [right_angle_condition]) (by\n    norm_num [circumradius_BXC_eq]\n    nlinarith)\n  norm_num at hc\n  field_simp at hc\n  nlinarith",
        "counterexample_rectangle_width_30_point_24_12_exposes_missing_factor_two",
    ),
    proof(
        "robustpa_brumo_2025_3", "length_AI", "negative",
        "by\n  intro h\n  simp only [point_A, point_I, Prod.dist_eq, Real.dist_eq, sub_zero] at h\n  have hs3 : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hs13 : (Real.sqrt (13 : ℝ)) ^ 2 = 13 := Real.sq_sqrt (by norm_num)\n  have hp3 : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hb3 : Real.sqrt (3 : ℝ) / 2 ≤ 7 / 2 := by nlinarith\n  rw [abs_of_nonneg (by norm_num), abs_of_nonneg (by positivity), max_eq_left hb3] at h\n  nlinarith",
        "product_metric_distance_is_seven_halves_not_euclidean_sqrt_thirteen",
    ),
    proof(
        "robustpa_cmimc_2025_7", "sum_first_n_integers", "positive",
        "by\n  calc\n    (Finset.sum (Finset.range n) fun i => i + 1) = (Finset.sum (Finset.range n) fun i => i) + n := by\n      rw [Finset.sum_add_distrib]\n      simp\n    _ = Finset.sum (Finset.range (n + 1)) (fun i => i) := by\n      rw [Finset.sum_range_succ]\n    _ = n * (n + 1) / 2 := by\n      simpa [Nat.mul_comm] using Finset.sum_range_id (n := n + 1)",
        "shift_sum_to_standard_sum_range_identity_without_reintroducing_parameters",
    ),
    proof(
        "robustpa_MATH_500_test_intermediate_algebra_960_json", "triangle_ineq_equiv", "positive",
        "by\n  simp [is_triangle_bool, Bool.and_eq_true]\n  omega",
        "unfold_boolean_triangle_test_and_split_max_cases_in_existing_context",
    ),
    proof(
        "robustpa_MATH_500_test_counting_and_probability_430_json", "bags_original_iff", "positive",
        "by\n  simp only [colors, Finset.mem_insert, Finset.mem_singleton] at hX hY\n  rcases hX with hX|hX|hX|hX|hX <;> rcases hY with hY|hY|hY|hY|hY <;>\n    subst X <;> subst Y <;> simp only [alice_final, bob_final, initial_bag, colors] <;> decide",
        "enumerate_color_pairs_in_existing_parameter_context",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  classical\n  rw [show removable_positions = (↑({(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)} : Finset (Fin 3 × Fin 3)) : Set (Fin 3 × Fin 3)) by\n    ext p\n    rcases p with ⟨i, j⟩\n    change Fin 3 at i\n    change Fin 3 at j\n    fin_cases i <;> fin_cases j <;> simp [removable_positions, start_pos, end_pos]]\n  change (↑({(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)} : Finset (Fin 3 × Fin 3)) : Set (Fin 3 × Fin 3)).ncard = 7\n  rw [Set.ncard_coe_finset]\n  norm_num",
        "enumerate_grid_positions_after_destructuring_the_defined_product_alias",
    ),
    proof(
        "robustpa_hmmt_feb_2025_29", "right_angle_circumradius_AXD_relation", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (208 : ℝ)) ^ 2 = 208 := Real.sq_sqrt (by norm_num)\n  have hp : 0 < Real.sqrt (208 : ℝ) := Real.sqrt_pos.2 (by norm_num)\n  have hc := h 26 8 12 (by norm_num [point_X_inside]) (by norm_num [right_angle_condition]) (by norm_num [circumradius_AXD_eq])\n  norm_num at hc\n  field_simp at hc\n  nlinarith",
        "counterexample_AXD_with_normalizer_closing_circumradius_premise",
    ),
    proof(
        "robustpa_hmmt_feb_2025_29", "right_angle_circumradius_BXC_relation", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (180 : ℝ)) ^ 2 = 180 := Real.sq_sqrt (by norm_num)\n  have hp : 0 < Real.sqrt (180 : ℝ) := Real.sqrt_pos.2 (by norm_num)\n  have hc := h 30 24 12 (by norm_num [point_X_inside]) (by norm_num [right_angle_condition]) (by norm_num [circumradius_BXC_eq])\n  norm_num at hc\n  field_simp at hc\n  nlinarith",
        "counterexample_BXC_with_normalizer_closing_circumradius_premise",
    ),
    proof(
        "robustpa_brumo_2025_3", "length_AI", "negative",
        "by\n  intro h\n  simp only [point_A, point_I, Prod.dist_eq, Real.dist_eq, sub_zero] at h\n  have hs3 : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hs13 : (Real.sqrt (13 : ℝ)) ^ 2 = 13 := Real.sq_sqrt (by norm_num)\n  have hp3 : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hb3 : Real.sqrt (3 : ℝ) / 2 ≤ 7 / 2 := by nlinarith\n  have hx : |(0 : ℝ) - 7 / 2| = 7 / 2 := by norm_num\n  have hy : |(0 : ℝ) - Real.sqrt 3 / 2| = Real.sqrt 3 / 2 := by\n    rw [abs_of_nonpos (by positivity)]\n    ring\n  rw [hx, hy, max_eq_left hb3] at h\n  nlinarith",
        "normalize_both_absolute_coordinates_of_product_metric",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  classical\n  rw [show removable_positions = (↑({(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)} : Finset (Fin 3 × Fin 3)) : Set (Fin 3 × Fin 3)) by\n    ext p\n    rcases p with ⟨i, j⟩\n    change Fin 3 at i\n    change Fin 3 at j\n    fin_cases i <;> fin_cases j <;> simp only [removable_positions, start_pos, end_pos, Set.mem_setOf_eq] <;> decide]\n  change (↑({(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)} : Finset (Fin 3 × Fin 3)) : Set (Fin 3 × Fin 3)).ncard = 7\n  rw [Set.ncard_coe_finset]\n  decide",
        "decide_each_of_nine_explicit_grid_membership_cases",
    ),
    proof(
        "robustpa_brumo_2025_3", "length_AI", "negative",
        "by\n  intro h\n  simp only [point_A, point_I, Prod.dist_eq, Real.dist_eq, sub_zero] at h\n  have hs3 : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hs13 : (Real.sqrt (13 : ℝ)) ^ 2 = 13 := Real.sq_sqrt (by norm_num)\n  have hp3 : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hb3 : Real.sqrt (3 : ℝ) / 2 ≤ 7 / 2 := by nlinarith\n  have hx : |(0 : ℝ) - 7 / 2| = 7 / 2 := by norm_num\n  have hy : |(0 : ℝ) - Real.sqrt 3 / 2| = Real.sqrt 3 / 2 := by\n    rw [abs_of_nonpos (by nlinarith)]\n    ring\n  rw [hx, hy, max_eq_left hb3] at h\n  nlinarith",
        "normalize_product_metric_coordinates_using_explicit_sqrt_nonnegativity",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  rw [show removable_positions = (↑({(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)} : Finset (Fin 3 × Fin 3)) : Set (Fin 3 × Fin 3)) by\n    ext p\n    rcases p with ⟨i, j⟩\n    change Fin 3 at i\n    change Fin 3 at j\n    fin_cases i <;> fin_cases j <;> simp [removable_positions, start_pos, end_pos]]\n  change (↑({(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)} : Finset (Fin 3 × Fin 3)) : Set (Fin 3 × Fin 3)).ncard = 7\n  rw [Set.ncard_coe_finset]\n  norm_num",
        "enumerate_grid_without_installing_classical_prop_decidability",
    ),
    proof(
        "robustpa_cmimc_2025_7", "nu_3_at_multiples_of_9", "negative",
        "by\n  intro h\n  have hc := h 1 (by norm_num)\n  norm_num [nu_3, a, last_digit, padicValNat] at hc",
        "counterexample_k_one_has_three_adic_valuation_two",
    ),
    proof(
        "robustpa_hmmt_feb_2025_6", "M_mod_2017", "negative",
        "by\n  decide",
        "kernel_evaluation_of_factorial_quotient_modulus_sign_error",
    ),
    proof(
        "robustpa_aime_2025_15", "cube_div_by_37_iff_div_by_27", "positive",
        "by\n  have hp : Nat.Prime 3 := by norm_num\n  rw [show divisor = 3^7 by norm_num [divisor], show 27 = 3^3 by norm_num]\n  rw [hp.pow_dvd_iff_le_factorization (pow_ne_zero 3 hx.ne')]\n  rw [Nat.factorization_pow]\n  change 7 ≤ 3 * x.factorization 3 ↔ 3^3 ∣ x\n  rw [hp.pow_dvd_iff_le_factorization hx.ne']\n  omega",
        "compare_three_adic_factorization_exponents",
    ),
    proof(
        "robustpa_aime_2025_15", "all_div_27_implies_sum_div_37", "positive",
        "by\n  rcases ha with ⟨a, rfl⟩\n  rcases hb with ⟨b, rfl⟩\n  rcases hc with ⟨c, rfl⟩\n  refine ⟨9 * (a^3 + b^3 + c^3), ?_⟩\n  norm_num [divisor]\n  ring",
        "factor_each_twenty_seven_multiple_cube_by_three_to_the_seventh",
    ),
    proof(
        "robustpa_hmmt_feb_2025_12", "removable_count", "positive",
        "by\n  classical\n  letI : Fintype GridPos := Fintype.ofEquiv (Fin 3 × Fin 3) (Equiv.refl _)\n  have hne : start_pos ≠ end_pos := by\n    intro h\n    have hv := congrArg (fun p : GridPos => p.1.val) h\n    norm_num [start_pos, end_pos] at hv\n  have heq : removable_positions = ({start_pos, end_pos} : Set GridPos)ᶜ := by\n    ext p\n    simp [removable_positions]\n  rw [heq, Set.ncard_compl, Set.ncard_pair hne, Nat.card_eq_fintype_card]\n  change Fintype.card (Fin 3 × Fin 3) - 2 = 7\n  norm_num",
        "install_explicit_fintype_for_defined_grid_alias_and_count_complement",
    ),
    proof(
        "robustpa_cmimc_2025_7", "nu_3_at_multiples_of_9", "negative",
        "by\n  intro h\n  have hc := h 1 (by norm_num)\n  have ha : a 9 = 3^2 * 13717421 := by norm_num [a, last_digit]\n  rw [ha, nu_3, padicValNat_base_pow_mul (by norm_num) (by norm_num)] at hc\n  rw [padicValNat.eq_zero_of_not_dvd (by norm_num)] at hc\n  norm_num at hc",
        "factor_a_nine_as_nine_times_a_nonmultiple_of_three",
    ),
    proof(
        "robustpa_hmmt_feb_2025_6", "M_mod_2017", "negative",
        "by\n  set_option maxRecDepth 100000 in\n    decide",
        "kernel_evaluate_factorial_quotient_with_sufficient_recursion_depth",
    ),
    proof(
        "robustpa_aime_2025_15", "triples_all_div_27_count", "negative",
        "by\n  intro h\n  let T := {t : ℕ × ℕ × ℕ | 27 ∣ t.1 ∧ t.1 ≤ bound ∧ 27 ∣ t.2.1 ∧ t.2.1 ≤ bound ∧ 27 ∣ t.2.2 ∧ t.2.2 ≤ bound}\n  let g : T → (Fin 730 × Fin 730 × Fin 730) := fun t =>\n    (⟨t.val.1, by simpa [T, bound] using Nat.lt_succ_of_le t.prop.2.1⟩,\n     ⟨t.val.2.1, by simpa [T, bound] using Nat.lt_succ_of_le t.prop.2.2.2.1⟩,\n     ⟨t.val.2.2, by simpa [T, bound] using Nat.lt_succ_of_le t.prop.2.2.2.2.2⟩)\n  have hg : Function.Injective g := by\n    intro x y he\n    apply Subtype.ext\n    apply Prod.ext\n    · exact congrArg (fun z => z.1.val) he\n    · apply Prod.ext\n      · exact congrArg (fun z => z.2.1.val) he\n      · exact congrArg (fun z => z.2.2.val) he\n  letI : Finite T := Finite.of_injective g hg\n  let f : (Fin 28 × Fin 28 × Fin 28) → T := fun k =>\n    ⟨(27 * k.1.val, 27 * k.2.1.val, 27 * k.2.2.val), by\n      simp only [T, bound]\n      constructor\n      · exact dvd_mul_right 27 k.1.val\n      constructor\n      · have := k.1.isLt\n        norm_num at *\n        omega\n      constructor\n      · exact dvd_mul_right 27 k.2.1.val\n      constructor\n      · have := k.2.1.isLt\n        norm_num at *\n        omega\n      constructor\n      · exact dvd_mul_right 27 k.2.2.val\n      · have := k.2.2.isLt\n        norm_num at *\n        omega⟩\n  have hf : Function.Injective f := by\n    intro x y he\n    apply Prod.ext\n    · have hv := congrArg (fun z => z.val.1) he\n      simp only [f] at hv\n      exact Fin.ext (by omega)\n    · apply Prod.ext\n      · have hv := congrArg (fun z => z.val.2.1) he\n        simp only [f] at hv\n        exact Fin.ext (by omega)\n      · have hv := congrArg (fun z => z.val.2.2) he\n        simp only [f] at hv\n        exact Fin.ext (by omega)\n  have hc := Nat.card_le_card_of_injective f hf\n  change Nat.card T = 27^3 at h\n  rw [h] at hc\n  norm_num at hc",
        "inject_twenty_eight_cubed_zero_inclusive_multiples_into_the_claimed_set",
    ),
    proof(
        "robustpa_aime_2025_15", "equiv_all_div_27", "negative",
        "by\n  intro h\n  have hc := h 1 3 314 (by norm_num) (by norm_num [bound]) (by norm_num) (by norm_num [bound]) (by norm_num) (by norm_num [bound])\n  norm_num [divisor] at hc",
        "counterexample_one_three_three_hundred_fourteen_has_cube_sum_divisible",
    ),
    proof(
        "robustpa_MATH_500_test_counting_and_probability_430_json",
        "robustpa_qwen3_8b_math_verify_MATH_500_robustpa_MATH_500_test_counting_and_probability_430_json",
        "positive",
        "by\n  norm_num [event_probability, num_colors, bob_bag_size_after_alice]",
        "evaluate_the_closed_rational_probability_definition",
    ),
    proof(
        "robustpa_cmimc_2025_34", "max_area_config", "positive",
        "by\n  dsimp\n  simp only [valid_hexagon_ordering]\n  constructor\n  · constructor\n    · ext p\n      simp only [Set.mem_range, hexagon_points, Set.mem_insert_iff, Set.mem_singleton_iff]\n      constructor\n      · rintro ⟨i, rfl⟩\n        fin_cases i <;> simp\n      · intro h\n        rcases h with h|h|h|h|h|h <;> subst p\n        all_goals first | exact ⟨0, rfl⟩ | exact ⟨1, rfl⟩ | exact ⟨2, rfl⟩ | exact ⟨3, rfl⟩ | exact ⟨4, rfl⟩ | exact ⟨5, rfl⟩\n    · constructor\n      · intro i j h\n        fin_cases i <;> fin_cases j <;> simp_all\n      · simp only [is_simple_hexagon, segments_intersect_interior]\n        repeat' apply And.intro\n        all_goals\n          rintro ⟨t, s, ht0, ht1, hs0, hs1, hx, hy⟩\n          simp at hx hy\n          try linarith\n  · simp [shoelace_area, next_index, Fin.sum_univ_succ]\n    norm_num",
        "enumerate_vertices_and_verify_all_ten_nonintersection_constraints",
    ),
    proof(
        "robustpa_cmimc_2025_34", "min_area_config", "negative",
        "by\n  intro h\n  rcases h with ⟨hv, ha⟩\n  have hs := hv.2.2\n  rcases hs with ⟨h1,h2,h3,h4,h5,h6,h7,h8,h9,h10⟩\n  apply h4\n  refine ⟨5/8, 1/16, by norm_num, by norm_num, by norm_num, by norm_num, ?_, ?_⟩ <;> simp <;> norm_num",
        "explicit_crossing_between_edges_one_two_and_three_four",
    ),
    *[
        proof(
            "robustpa_cmimc_2025_25", name, "positive",
            "by\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [Q1, P1, P2, Prod.dist_eq, Real.dist_eq]\n  rw [max_lt_iff]\n  constructor <;> rw [abs_lt] <;> constructor <;> nlinarith",
            "bound_both_coordinates_of_the_product_metric_distance",
        )
        for name in ["distance_Q1_P1_lt", "distance_Q1_P2_lt"]
    ],
    proof(
        "robustpa_cmimc_2025_25", "P1_on_E_BC", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [ellipse_E_BC, P1, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  simp only [sub_zero] at h\n  have h1x : |(17 - 12*Real.sqrt 3)/26 - 1| = 1 - (17 - 12*Real.sqrt 3)/26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  have h1y : |(12 + 3*Real.sqrt 3)/26| = (12 + 3*Real.sqrt 3)/26 := by rw [abs_of_nonneg (by nlinarith)]\n  have h2x : |(17 - 12*Real.sqrt 3)/26 - 1/2| = 1/2 - (17 - 12*Real.sqrt 3)/26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  have h2y : |(12 + 3*Real.sqrt 3)/26 - Real.sqrt 3/2| = Real.sqrt 3/2 - (12 + 3*Real.sqrt 3)/26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  rw [h1x,h1y,h2x,h2y, max_eq_left (by nlinarith), max_eq_left (by nlinarith)] at h\n  nlinarith",
        "evaluate_P1_two_sup_distances_to_B_and_C",
    ),
    *[
        proof(
            "robustpa_cmimc_2025_25", name, "negative",
            f"by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [{ellipse}, Q1, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  simp only [sub_zero] at h\n  have h1x : |(1/2:ℝ)|=1/2 := by norm_num\n  have h1y : |(24+7*Real.sqrt 3)/26|=(24+7*Real.sqrt 3)/26 := by rw [abs_of_nonneg (by nlinarith)]\n  have h2x : |(1/2:ℝ)-1/2|=0 := by norm_num\n  have h2y : |(24+7*Real.sqrt 3)/26-Real.sqrt 3/2|=(24+7*Real.sqrt 3)/26-Real.sqrt 3/2 := by rw [abs_of_nonneg (by nlinarith)]\n  rw [h1x,h1y,h2x,h2y, max_eq_right (by nlinarith), max_eq_right (by nlinarith)] at h\n  nlinarith",
            "evaluate_Q1_sup_metric_ellipse_sum",
        )
        for name, ellipse in [("Q1_on_E_BC", "ellipse_E_BC"), ("Q1_on_E_AC", "ellipse_E_AC")]
    ],
    proof(
        "robustpa_cmimc_2025_25", "P2_on_E_AB", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [ellipse_E_AB, P2, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  simp only [sub_zero] at h\n  have hx0 : |(17+12*Real.sqrt 3)/26|=(17+12*Real.sqrt 3)/26 := by rw [abs_of_nonneg (by nlinarith)]\n  have hy : |(3*Real.sqrt 3-12)/26|=(12-3*Real.sqrt 3)/26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  have hx1 : |(17+12*Real.sqrt 3)/26-1|=(17+12*Real.sqrt 3)/26-1 := by rw [abs_of_nonneg (by nlinarith)]\n  rw [hx0,hy,hx1,max_eq_left (by nlinarith),max_eq_left (by nlinarith)] at h\n  nlinarith",
        "evaluate_P2_sup_distances_to_A_and_B",
    ),
    proof(
        "robustpa_cmimc_2025_25", "P2_on_E_AC", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [ellipse_E_AC, P2, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  simp only [sub_zero] at h\n  have hx0 : |(17+12*Real.sqrt 3)/26|=(17+12*Real.sqrt 3)/26 := by rw [abs_of_nonneg (by nlinarith)]\n  have hy0 : |(3*Real.sqrt 3-12)/26|=(12-3*Real.sqrt 3)/26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  have hx1 : |(17+12*Real.sqrt 3)/26-1/2|=(17+12*Real.sqrt 3)/26-1/2 := by rw [abs_of_nonneg (by nlinarith)]\n  have hy1 : |(3*Real.sqrt 3-12)/26-Real.sqrt 3/2|=Real.sqrt 3/2-(3*Real.sqrt 3-12)/26 := by rw [abs_of_nonpos (by nlinarith)]; ring\n  rw [hx0,hy0,hx1,hy1,max_eq_left (by nlinarith),max_eq_right (by nlinarith)] at h\n  nlinarith",
        "evaluate_P2_sup_distances_to_A_and_C",
    ),
    proof(
        "robustpa_cmimc_2025_25", "triangle_GHI_nonintersecting", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [triangle_edges_nonintersecting] at h\n  rcases h with ⟨h1,h2,h3,h4,h5,h6,h7,h8,h9⟩\n  apply h9 ((24 + 7 * Real.sqrt 3) / 48) (1/2) <;> try nlinarith\n  simp only [P2, P1]\n  constructor <;> nlinarith",
        "explicit_intersection_of_P2P1_with_the_CA_edge",
    ),
    proof(
        "robustpa_cmimc_2025_25", "Q1_on_E_BC", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [ellipse_E_BC, Q1, Set.mem_setOf_eq, Prod.dist_eq, Real.dist_eq] at h\n  simp only [sub_zero] at h\n  have h1x : |(1/2:ℝ)-1|=1/2 := by norm_num\n  have h1y : |(24+7*Real.sqrt 3)/26|=(24+7*Real.sqrt 3)/26 := by rw [abs_of_nonneg (by nlinarith)]\n  have h2x : |(1/2:ℝ)-1/2|=0 := by norm_num\n  have h2y : |(24+7*Real.sqrt 3)/26-Real.sqrt 3/2|=(24+7*Real.sqrt 3)/26-Real.sqrt 3/2 := by rw [abs_of_nonneg (by nlinarith)]\n  rw [h1x,h1y,h2x,h2y, max_eq_right (by nlinarith), max_eq_right (by nlinarith)] at h\n  nlinarith",
        "evaluate_Q1_distances_to_B_and_C_in_sup_metric",
    ),
    proof(
        "robustpa_cmimc_2025_25", "triangle_GHI_nonintersecting", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hl : 1 < Real.sqrt (3 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3 : ℝ) < 2 := by nlinarith\n  simp only [triangle_edges_nonintersecting] at h\n  rcases h with ⟨h1,h2,h3,h4,h5,h6,h7,h8,h9⟩\n  apply h9 ((24 + 7 * Real.sqrt 3) / 48) (1/2) <;> try nlinarith\n  simp only [P2, P1]\n  apply Prod.ext <;> nlinarith",
        "prove_coordinate_equality_at_P2P1_CA_intersection",
    ),
    proof(
        "robustpa_brumo_2025_22", "counterexample_exists", "negative",
        "by\n  intro h\n  have hp := h.1.1.2.2 9\n  rcases hp with ⟨p, hp⟩\n  have hb := h.1.1.1 p.1 p.2\n  omega",
        "finite_grid_map_cannot_be_surjective_onto_nat_at_value_nine",
    ),
    proof(
        "robustpa_brumo_2025_22", "magic_square_count", "negative",
        "by\n  intro h\n  rcases h with ⟨S, hcard, hchar⟩\n  have hempty : S = ∅ := by\n    ext g\n    simp only [Finset.notMem_empty, iff_false]\n    intro hg\n    have hm := (hchar g).mp hg\n    have hp := hm.1.1.2.2 9\n    rcases hp with ⟨p, hp⟩\n    have hb := hm.1.1.1 p.1 p.2\n    omega\n  rw [hempty] at hcard\n  norm_num at hcard",
        "all_magic_squares_are_empty_because_bijective_target_was_nat",
    ),
    proof(
        "robustpa_brumo_2025_22", "counterexample_exists", "negative",
        "by\n  intro h\n  simp only [is_valid_grid, is_permutation_1_to_9, Function.Bijective] at h\n  obtain ⟨p, hp⟩ := h.1.1.2.2 9\n  have hb := h.1.1.1 p.1 p.2\n  omega",
        "unfold_invalid_surjectivity_of_nine_grid_cells_onto_nat",
    ),
    proof(
        "robustpa_brumo_2025_22", "magic_square_count", "negative",
        "by\n  intro h\n  rcases h with ⟨S, hcard, hchar⟩\n  have hempty : S = ∅ := by\n    ext g\n    simp only [Finset.notMem_empty, iff_false]\n    intro hg\n    have hm := (hchar g).mp hg\n    simp only [is_magic_square, is_valid_grid, is_permutation_1_to_9, Function.Bijective] at hm\n    obtain ⟨p, hp⟩ := hm.1.1.2.2 9\n    have hb := hm.1.1.1 p.1 p.2\n    omega\n  rw [hempty] at hcard\n  norm_num at hcard",
        "unfold_and_show_the_magic_square_set_is_empty",
    ),
    proof(
        "robustpa_brumo_2025_12", "base_case_1", "positive",
        "by\n  simp [count_valid_permutations, a1, is_valid_permutation, valid_assignment]",
        "compute_the_unique_permutation_on_fin_one",
    ),
    proof(
        "robustpa_brumo_2025_12", "base_case_2", "positive",
        "by\n  rw [count_well_defined]\n  have hall : ∀ σ : Equiv.Perm (Fin 2), is_valid_permutation 2 σ := by\n    intro σ i\n    fin_cases i\n    · have hv := (σ 0).isLt\n      interval_cases h : (σ 0).val <;> simp [is_valid_permutation, valid_assignment, h]\n    · have hv := (σ 1).isLt\n      interval_cases h : (σ 1).val <;> simp [is_valid_permutation, valid_assignment, h]\n  rw [show {σ : Equiv.Perm (Fin 2) | is_valid_permutation 2 σ} = Set.univ by ext σ; simp [hall σ]]\n  simp [Nat.card_eq_fintype_card, Fintype.card_perm, a2]",
        "enumerate_outputs_and_count_the_two_permutations_on_fin_two",
    ),
    proof(
        "robustpa_brumo_2025_12", "recurrence_holds", "negative",
        "by\n  intro hr\n  have hall2 : ∀ σ : Equiv.Perm (Fin 2), is_valid_permutation 2 σ := by\n    intro σ i\n    fin_cases i\n    · have hv := (σ 0).isLt\n      interval_cases h : (σ 0).val <;> simp [is_valid_permutation, valid_assignment, h]\n    · have hv := (σ 1).isLt\n      interval_cases h : (σ 1).val <;> simp [is_valid_permutation, valid_assignment, h]\n  have hall3 : ∀ σ : Equiv.Perm (Fin 3), is_valid_permutation 3 σ := by\n    intro σ i\n    fin_cases i\n    · have hv := (σ 0).isLt\n      interval_cases h : (σ 0).val <;> simp [is_valid_permutation, valid_assignment, h]\n    · have hv := (σ 1).isLt\n      interval_cases h : (σ 1).val <;> simp [is_valid_permutation, valid_assignment, h]\n    · have hv := (σ 2).isLt\n      interval_cases h : (σ 2).val <;> simp [is_valid_permutation, valid_assignment, h]\n  have hc1 : count_valid_permutations 1 = 1 := by\n    simp [count_valid_permutations, is_valid_permutation, valid_assignment, Nat.card_eq_fintype_card, Fintype.card_perm]\n  have hc2 : count_valid_permutations 2 = 2 := by\n    rw [count_valid_permutations]\n    rw [show {σ : Equiv.Perm (Fin 2) | is_valid_permutation 2 σ} = Set.univ by ext σ; simp [hall2 σ]]\n    simp [Nat.card_eq_fintype_card, Fintype.card_perm]\n  have hc3 : count_valid_permutations 3 = 6 := by\n    rw [count_valid_permutations]\n    rw [show {σ : Equiv.Perm (Fin 3) | is_valid_permutation 3 σ} = Set.univ by ext σ; simp [hall3 σ]]\n    simp [Nat.card_eq_fintype_card, Fintype.card_perm]\n    norm_num [Nat.factorial]\n  have hc := hr 3 (by norm_num)\n  rw [hc1, hc2, hc3] at hc\n  norm_num at hc",
        "counterexample_n_three_counts_all_six_permutations_not_four",
    ),
    proof(
        "robustpa_brumo_2025_22", "counterexample_exists", "negative",
        "by\n  intro h\n  simp only [is_valid_grid, is_permutation_1_to_9, Function.Bijective] at h\n  obtain ⟨p, hp⟩ := h.1.1.2.2 9\n  have hb := h.1.1.1 p.1 p.2\n  have heq : counterexample_grid p.1 p.2 = 10 := by\n    calc\n      counterexample_grid p.1 p.2 = (counterexample_grid p.1 p.2 - 1) + 1 := (Nat.sub_add_cancel hb.1).symm\n      _ = 9 + 1 := by rw [hp]\n      _ = 10 := by norm_num\n  omega",
        "turn_surjectivity_equation_into_forbidden_grid_value_ten",
    ),
    proof(
        "robustpa_brumo_2025_22", "magic_square_count", "negative",
        "by\n  intro h\n  rcases h with ⟨S, hcard, hchar⟩\n  have hempty : S = ∅ := by\n    ext g\n    simp only [Finset.notMem_empty, iff_false]\n    intro hg\n    have hm := (hchar g).mp hg\n    simp only [is_magic_square, is_valid_grid, is_permutation_1_to_9, Function.Bijective] at hm\n    obtain ⟨p, hp⟩ := hm.1.1.2.2 9\n    have hb := hm.1.1.1 p.1 p.2\n    have heq : g p.1 p.2 = 10 := by\n      calc\n        g p.1 p.2 = (g p.1 p.2 - 1) + 1 := (Nat.sub_add_cancel hb.1).symm\n        _ = 9 + 1 := by rw [hp]\n        _ = 10 := by norm_num\n    omega\n  rw [hempty] at hcard\n  norm_num at hcard",
        "derive_value_ten_then_empty_the_magic_square_finset",
    ),
    proof(
        "robustpa_hmmt_feb_2025_25", "circumradius_equilateral_6", "negative",
        "by\n  intro h\n  have hs : (Real.sqrt (3 : ℝ)) ^ 2 = 3 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3 : ℝ) := Real.sqrt_nonneg 3\n  have hR3 : 3 < 2 * Real.sqrt (3 : ℝ) := by nlinarith\n  have hR4 : 2 * Real.sqrt (3 : ℝ) < 4 := by nlinarith\n  have heq : equilateral_triangle_ABC ((0,0) : ℝ × ℝ) (6,0) (3,6) := by\n    norm_num [equilateral_triangle_ABC, Prod.dist_eq, Real.dist_eq]\n  rcases h (0,0) (6,0) (3,6) heq with ⟨O,R,rfl,hA,hB,hC⟩\n  rcases O with ⟨x,y⟩\n  simp only [Prod.dist_eq, Real.dist_eq] at hA hB hC\n  simp only [sub_zero] at hA hB\n  have hAx : |x| ≤ 2 * Real.sqrt 3 := le_trans (le_max_left _ _) (le_of_eq hA)\n  have hAy : |y| ≤ 2 * Real.sqrt 3 := le_trans (le_max_right _ _) (le_of_eq hA)\n  have hBx : |x - 6| ≤ 2 * Real.sqrt 3 := le_trans (le_max_left _ _) (le_of_eq hB)\n  have hxlo : 6 - 2 * Real.sqrt 3 ≤ x := by rw [abs_le] at hBx; linarith\n  have hxhi : x ≤ 2 * Real.sqrt 3 := by rw [abs_le] at hAx; linarith\n  have hyhi : y ≤ 2 * Real.sqrt 3 := by rw [abs_le] at hAy; linarith\n  have hxc : |x - 3| < 2 * Real.sqrt 3 := by\n    rw [abs_lt]\n    constructor <;> nlinarith\n  have hy6 : |y - 6| = 2 * Real.sqrt 3 := by\n    rcases max_choice |x-3| |y-6| with hm|hm\n    · rw [hm] at hC\n      nlinarith\n    · rw [hm] at hC\n      exact hC\n  have hy : y = 6 - 2 * Real.sqrt 3 := by\n    rw [abs_of_nonpos (by nlinarith)] at hy6\n    linarith\n  have hyabs : |y| < 2 * Real.sqrt 3 := by\n    rw [hy, abs_of_nonneg (by nlinarith)]\n    nlinarith\n  have hxabs : |x| = 2 * Real.sqrt 3 := by\n    by_contra hn\n    have hxl : |x| < 2 * Real.sqrt 3 := lt_of_le_of_ne hAx hn\n    have hm := max_lt hxl hyabs\n    rw [hA] at hm\n    exact (lt_irrefl _ hm)\n  have hxbabs : |x - 6| = 2 * Real.sqrt 3 := by\n    by_contra hn\n    have hxl : |x - 6| < 2 * Real.sqrt 3 := lt_of_le_of_ne hBx hn\n    have hm := max_lt hxl hyabs\n    rw [hB] at hm\n    exact (lt_irrefl _ hm)\n  rw [abs_of_nonneg (by nlinarith)] at hxabs\n  rw [abs_of_nonpos (by nlinarith)] at hxbabs\n  nlinarith",
        "L_infinity_equilateral_counterexample_has_no_claimed_radius_center",
    ),
    proof(
        "robustpa_brumo_2025_22", "counterexample_exists", "negative",
        "by\n  intro h\n  simp only [is_valid_grid, is_permutation_1_to_9, Function.Bijective] at h\n  obtain ⟨p, hp⟩ := h.1.1.2.2 9\n  have hb := h.1.1.1 p.1 p.2\n  dsimp only at hp\n  omega",
        "simplify_surjective_lambda_application_then_contradict_bound",
    ),
    proof(
        "robustpa_brumo_2025_22", "magic_square_count", "negative",
        "by\n  intro h\n  rcases h with ⟨S, hcard, hchar⟩\n  have hempty : S = ∅ := by\n    ext g\n    simp only [Finset.notMem_empty, iff_false]\n    intro hg\n    have hm := (hchar g).mp hg\n    simp only [is_magic_square, is_valid_grid, is_permutation_1_to_9, Function.Bijective] at hm\n    obtain ⟨p, hp⟩ := hm.1.1.2.2 9\n    have hb := hm.1.1.1 p.1 p.2\n    dsimp only at hp\n    omega\n  rw [hempty] at hcard\n  norm_num at hcard",
        "empty_magic_square_finset_after_simplifying_lambda_application",
    ),
    proof(
        "robustpa_aime_2025_11", "claimed_values_are_intersections", "negative",
        "by\n  intro h\n  let y : ℝ := (-1 + Real.sqrt 3097) / 68\n  have hy := h y (by simp [y, claimed_positive_y_values])\n  have hs : (Real.sqrt (3097 : ℝ)) ^ 2 = 3097 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3097 : ℝ) := Real.sqrt_nonneg 3097\n  have hl : 55 < Real.sqrt (3097 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3097 : ℝ) < 56 := by nlinarith\n  have hfloor : ⌊(34 * y^2 + 1) / 4⌋ = (5 : ℤ) := by\n    rw [Int.floor_eq_iff]\n    constructor <;> norm_num [y] <;> nlinarith\n  simp only [is_intersection_y, parabola_x, f] at hy\n  rw [hfloor] at hy\n  norm_num only [Int.cast_ofNat, mul_comm (4:ℝ) 5] at hy\n  have hn : ¬(-1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 1) := by\n    intro hh\n    nlinarith [hs]\n  have hb : 1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 3 := by\n    constructor <;> dsimp [y] <;> nlinarith\n  rw [if_neg hn, if_pos hb] at hy\n  dsimp [y] at hy\n  nlinarith",
        "explicit_bad_claimed_value_with_sqrt_3097_falls_in_wrong_branch",
    ),
    proof(
        "robustpa_aime_2025_11", "claimed_values_are_intersections", "negative",
        "by\n  intro h\n  let y : ℝ := (-1 + Real.sqrt 3097) / 68\n  have hy := h y (by simp [y, claimed_positive_y_values])\n  have hs : (Real.sqrt (3097 : ℝ)) ^ 2 = 3097 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3097 : ℝ) := Real.sqrt_nonneg 3097\n  have hl : 55 < Real.sqrt (3097 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3097 : ℝ) < 56 := by nlinarith\n  have hfloor : ⌊(34 * y^2 + 1) / 4⌋ = (5 : ℤ) := by\n    rw [Int.floor_eq_iff]\n    constructor <;> norm_num [y] <;> nlinarith\n  simp only [is_intersection_y, parabola_x, f] at hy\n  rw [hfloor] at hy\n  norm_num only [Int.cast_ofNat, mul_comm (4:ℝ) 5] at hy\n  dsimp only [f_base] at hy\n  have hn : ¬(-1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 1) := by\n    dsimp [y]\n    constructor\n    intro hh\n    nlinarith\n  have hb : 1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 3 := by\n    constructor <;> dsimp [y] <;> nlinarith\n  rw [if_neg hn, if_pos hb] at hy\n  dsimp [y] at hy\n  nlinarith",
        "unfold_piecewise_base_before_selecting_the_bad_value_branch",
    ),
    proof(
        "robustpa_aime_2025_11", "claimed_values_are_intersections", "negative",
        "by\n  intro h\n  let y : ℝ := (-1 + Real.sqrt 3097) / 68\n  have hy := h y (by simp [y, claimed_positive_y_values])\n  have hs : (Real.sqrt (3097 : ℝ)) ^ 2 = 3097 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (3097 : ℝ) := Real.sqrt_nonneg 3097\n  have hl : 55 < Real.sqrt (3097 : ℝ) := by nlinarith\n  have hu : Real.sqrt (3097 : ℝ) < 56 := by nlinarith\n  have hfloor : ⌊(34 * y^2 + 1) / 4⌋ = (5 : ℤ) := by\n    rw [Int.floor_eq_iff]\n    constructor <;> norm_num [y] <;> nlinarith\n  simp only [is_intersection_y, parabola_x, f] at hy\n  rw [hfloor] at hy\n  norm_num only [Int.cast_ofNat, mul_comm (4:ℝ) 5] at hy\n  dsimp only [f_base] at hy\n  have hn : ¬(-1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 1) := by\n    dsimp [y]\n    intro hh\n    nlinarith\n  have hb : 1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 3 := by\n    constructor <;> dsimp [y] <;> nlinarith\n  rw [if_neg hn, if_pos hb] at hy\n  dsimp [y] at hy\n  nlinarith",
        "correctly_unfold_and_select_bad_sqrt_3097_branch",
    ),
    proof(
        "robustpa_cmimc_2025_5", "count_multiples_correct", "negative",
        "by\n  intro hbad\n  have heq : (Finset.filter (fun n => 0 < n ∧ n < 1000000 ∧ 77 ∣ n) (Finset.range 1000000)) =\n      Finset.filter (fun n => n ≠ 0 ∧ 77 ∣ n) (Finset.range (Nat.succ 999999)) := by\n    ext n\n    simp only [Finset.mem_filter, Finset.mem_range]\n    omega\n  rw [heq] at hbad\n  have hcorrect := Nat.card_multiples' 999999 77\n  norm_num at hcorrect\n  omega",
        "correct_multiple_count_is_12987_by_mathlib_card_multiples",
    ),
    proof(
        "robustpa_cmimc_2025_5", "count_multiples_correct", "negative",
        "by\n  intro hbad\n  have heq : (Finset.filter (fun n => 0 < n ∧ n < 1000000 ∧ 77 ∣ n) (Finset.range 1000000)) =\n      Finset.filter (fun n => n ≠ 0 ∧ 77 ∣ n) (Finset.range (Nat.succ 999999)) := by\n    ext n\n    simp only [Finset.mem_filter, Finset.mem_range]\n    omega\n  rw [heq] at hbad\n  have hcorrect := Nat.card_multiples' 999999 77\n  norm_num at hcorrect\n  rw [hcorrect] at hbad\n  norm_num at hbad",
        "rewrite_claim_with_the_12987_cardinality_theorem",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_no_other_real", "negative",
        "by\n  intro h\n  have hn := h 13 (by norm_num)\n  apply hn\n  refine ⟨(1 : ℂ), (1 : ℂ), ?_, ?_, ?_⟩\n  · norm_num [p2]\n  · norm_num [p2]\n  · refine ⟨(-1 : ℂ), 2, by norm_num, ?_, ?_, ?_, ?_⟩ <;>\n      rw [Complex.norm_def] <;> norm_num [Complex.normSq_apply]",
        "m_thirteen_is_in_U_using_the_repeated_root_one",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_excludes_nonreal", "negative",
        "by\n  intro h\n  let m : ℂ := -3 - 4 * Complex.I\n  have hn := h m (by norm_num [m])\n  apply hn\n  let z : ℂ := -1 + 2 * Complex.I\n  refine ⟨z, z, ?_, ?_, ?_⟩\n  · apply Complex.ext <;> norm_num [p2, m, z, pow_two, Complex.mul_re, Complex.mul_im]\n  · apply Complex.ext <;> norm_num [p2, m, z, pow_two, Complex.mul_re, Complex.mul_im]\n  · refine ⟨(-1 : ℂ), 2, by norm_num, ?_, ?_, ?_, ?_⟩ <;>\n      rw [Complex.norm_def] <;> norm_num [z, Complex.normSq_apply]",
        "nonreal_m_minus_three_minus_four_i_is_in_U_via_repeated_p1_root",
    ),
    proof(
        "robustpa_aime_2025_8", "tangent_condition", "negative",
        "by\n  intro h\n  have hr : distance_to_line (15/8 : ℝ) = 5 := by norm_num [distance_to_line]\n  rcases (h (15/8)).mpr hr with ⟨z, hz, hu⟩\n  rcases hz with ⟨hc, hb⟩\n  have hc2 := congrArg (fun x : ℝ => x^2) hc\n  have hb2 := congrArg (fun x : ℝ => x^2) hb\n  simp only [circle_condition, bisector_condition, Complex.sq_norm] at hc2 hb2\n  simp only [Complex.normSq_apply, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im] at hc2 hb2\n  norm_num at hc2 hb2\n  have hline : 8*z.re - 6*z.im = 22 := by nlinarith [hb2]\n  have hcs : (8*(z.re-25)-6*(z.im-20))^2 ≤ 100*((z.re-25)^2+(z.im-20)^2) := by\n    nlinarith [sq_nonneg (6*(z.re-25)+8*(z.im-20))]\n  nlinarith [hc2]",
        "k_fifteen_eighths_satisfies_wrong_distance_formula_but_actual_bisector_misses_circle",
    ),
    proof(
        "robustpa_hmmt_feb_2025_16", "sum_S_is_26600", "negative",
        "by\n  intro hbad\n  have htrue : sum_nonoverlapping_S = 12600 := by\n"
        "    rw [codex_sum_eq_B]\n"
        "    norm_num [codex_sumB, codex_allRects, codex_validB, Finset.sum_filter, Finset.sum_product, Finset.sum_range_succ, "
        + ", ".join(RECTANGLE_HELPER_NAMES) + "]\n  omega",
        "computable_mirror_and_225_kernel_checked_rectangle_counts_sum_to_12600",
        prefix_code=RECTANGLE_PREFIX,
    ),
    proof(
        "robustpa_brumo_2025_3", "inside_AFB", "negative",
        "by\n  intro h\n  have hA0 := h 0 (by norm_num)\n  have hA : point_A ∈ interior triangle_AFB := by simpa [param_AI, point_A] using hA0\n  have hopen := Metric.isOpen_iff.1 isOpen_interior point_A hA\n  rcases hopen with ⟨ε, hε, hball⟩\n  let q : ℝ × ℝ := (-ε/2, 0)\n  have hqball : q ∈ Metric.ball point_A ε := by\n    simp only [Metric.mem_ball, q, point_A, Prod.dist_eq, Real.dist_eq]\n    rw [max_lt_iff]\n    constructor\n    · rw [abs_of_nonpos (by linarith)]\n      linarith\n    · norm_num\n  have hq := interior_subset (hball hqball)\n  have hsub : triangle_AFB ⊆ {p : ℝ × ℝ | 0 ≤ p.1} := by\n    apply convexHull_min\n    · simp only [Set.insert_subset_iff, Set.singleton_subset_iff, Set.mem_setOf_eq]\n      norm_num [point_A, point_F, point_B]\n    · change Convex ℝ ((LinearMap.fst ℝ ℝ ℝ) ⁻¹' Set.Ici 0)\n      exact (convex_Ici (0 : ℝ) : Convex ℝ (Set.Ici (0 : ℝ))).linear_preimage _\n  have := hsub hq\n  change 0 ≤ -ε/2 at this\n  linarith",
        "closed_interval_includes_t_zero_but_vertex_A_is_not_in_triangle_interior",
    ),
    proof(
        "robustpa_brumo_2025_3", "inside_AFB", "negative",
        "by\n  intro h\n  have hA0 := h 0 (by norm_num)\n  have hA : point_A ∈ interior triangle_AFB := by simpa [param_AI, point_A] using hA0\n  have hopen := Metric.isOpen_iff.1 isOpen_interior point_A hA\n  rcases hopen with ⟨ε, hε, hball⟩\n  let q : ℝ × ℝ := (-ε/2, 0)\n  have hqball : q ∈ Metric.ball point_A ε := by\n    simp only [Metric.mem_ball, q, point_A, Prod.dist_eq, Real.dist_eq]\n    rw [max_lt_iff]\n    constructor\n    · rw [abs_of_nonpos (by linarith)]\n      linarith\n    · simpa using hε\n  have hq := interior_subset (hball hqball)\n  have hsub : triangle_AFB ⊆ {p : ℝ × ℝ | 0 ≤ p.1} := by\n    apply convexHull_min\n    · simp only [Set.insert_subset_iff, Set.singleton_subset_iff, Set.mem_setOf_eq]\n      norm_num [point_A, point_F, point_B]\n    · change Convex ℝ ((LinearMap.fst ℝ ℝ ℝ) ⁻¹' Set.Ici 0)\n      exact (convex_Ici (0 : ℝ) : Convex ℝ (Set.Ici (0 : ℝ))).linear_preimage _\n  have := hsub hq\n  change 0 ≤ -ε/2 at this\n  linarith",
        "open_ball_at_vertex_contains_a_point_with_negative_first_coordinate",
    ),
    proof(
        "robustpa_brumo_2025_3", "inside_CHD", "negative",
        "by\n  intro h\n  have hp := interior_subset (h (1/2) (by norm_num))\n  have hsub : triangle_CHD ⊆ {p : ℝ × ℝ | 2 ≤ p.1} := by\n    apply convexHull_min\n    · simp only [Set.insert_subset_iff, Set.singleton_subset_iff, Set.mem_setOf_eq]\n      norm_num [point_C, point_H, point_D]\n    · change Convex ℝ ((LinearMap.fst ℝ ℝ ℝ) ⁻¹' Set.Ici 2)\n      exact (convex_Ici (2 : ℝ) : Convex ℝ (Set.Ici (2 : ℝ))).linear_preimage _\n  have := hsub hp\n  norm_num [param_AI] at this",
        "the_claimed_closed_interval_starts_at_a_point_left_of_the_entire_triangle",
    ),
    proof(
        "robustpa_brumo_2025_3", "inside_DIE", "negative",
        "by\n  intro h\n  have hI0 := h 1 (by norm_num)\n  have hI : point_I ∈ interior triangle_DIE := by simpa [param_AI, point_I] using hI0\n  rcases Metric.isOpen_iff.1 isOpen_interior point_I hI with ⟨ε, hε, hball⟩\n  let q : ℝ × ℝ := (7/2, Real.sqrt 3/2 + ε/2)\n  have hqball : q ∈ Metric.ball point_I ε := by\n    simp only [Metric.mem_ball, q, point_I, Prod.dist_eq, Real.dist_eq]\n    rw [max_lt_iff]\n    constructor\n    · norm_num\n    · rw [abs_of_nonpos (by linarith)]\n      linarith\n  have hq := interior_subset (hball hqball)\n  have hsub : triangle_DIE ⊆ {p : ℝ × ℝ | p.2 ≤ Real.sqrt 3/2} := by\n    apply convexHull_min\n    · simp only [Set.insert_subset_iff, Set.singleton_subset_iff, Set.mem_setOf_eq]\n      norm_num [point_D, point_I, point_E]\n      exact Real.sqrt_nonneg 3\n    · change Convex ℝ ((LinearMap.snd ℝ ℝ ℝ) ⁻¹' Set.Iic (Real.sqrt 3/2))\n      exact (convex_Iic (Real.sqrt 3/2) : Convex ℝ (Set.Iic (Real.sqrt 3/2))).linear_preimage _\n  have := hsub hq\n  change Real.sqrt 3/2 + ε/2 ≤ Real.sqrt 3/2 at this\n  linarith",
        "closed_interval_includes_vertex_I_which_is_not_in_the_triangle_interior",
    ),
    proof(
        "robustpa_brumo_2025_3", "inside_DIE", "negative",
        "by\n  intro h\n  have hI0 := h 1 (by norm_num)\n  have hI : point_I ∈ interior triangle_DIE := by simpa [param_AI, point_I] using hI0\n  rcases Metric.isOpen_iff.1 isOpen_interior point_I hI with ⟨ε, hε, hball⟩\n  let q : ℝ × ℝ := (7/2, Real.sqrt 3/2 + ε/2)\n  have hqball : q ∈ Metric.ball point_I ε := by\n    simp only [Metric.mem_ball, q, point_I, Prod.dist_eq, Real.dist_eq]\n    rw [max_lt_iff]\n    constructor\n    · simpa using hε\n    · rw [abs_of_nonneg (by linarith)]\n      linarith\n  have hq := interior_subset (hball hqball)\n  have hsub : triangle_DIE ⊆ {p : ℝ × ℝ | p.2 ≤ Real.sqrt 3/2} := by\n    apply convexHull_min\n    · simp only [Set.insert_subset_iff, Set.singleton_subset_iff, Set.mem_setOf_eq]\n      norm_num [point_D, point_I, point_E]\n      positivity\n    · change Convex ℝ ((LinearMap.snd ℝ ℝ ℝ) ⁻¹' Set.Iic (Real.sqrt 3/2))\n      exact (convex_Iic (Real.sqrt 3/2) : Convex ℝ (Set.Iic (Real.sqrt 3/2))).linear_preimage _\n  have := hsub hq\n  change Real.sqrt 3/2 + ε/2 ≤ Real.sqrt 3/2 at this\n  linarith",
        "open_ball_above_vertex_I_escapes_the_supporting_halfspace",
    ),
    proof(
        "robustpa_brumo_2025_3", "inside_BGC", "negative",
        "by\n  intro h\n  let p := param_AI (1/3)\n  have hp : p ∈ interior triangle_BGC := h (1/3) (by norm_num)\n  rcases Metric.isOpen_iff.1 isOpen_interior p hp with ⟨ε, hε, hball⟩\n  let q : ℝ × ℝ := (p.1, p.2 + ε/2)\n  have hqball : q ∈ Metric.ball p ε := by\n    simp only [Metric.mem_ball, q, Prod.dist_eq, Real.dist_eq]\n    rw [max_lt_iff]\n    constructor\n    · simpa using hε\n    · rw [abs_of_nonneg (by linarith)]\n      linarith\n  have hq := interior_subset (hball hqball)\n  let L : (ℝ × ℝ) →ₗ[ℝ] ℝ := {\n    toFun := fun z => Real.sqrt 3 * z.1 - z.2\n    map_add' := by intro x y; simp; ring\n    map_smul' := by intro c x; simp; ring }\n  have hsub : triangle_BGC ⊆ {z : ℝ × ℝ | Real.sqrt 3 ≤ L z} := by\n    apply convexHull_min\n    · simp only [Set.insert_subset_iff, Set.singleton_subset_iff, Set.mem_setOf_eq]\n      simp [L, point_B, point_G, point_C]\n      constructor\n      · ring\n      · have hs : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n        nlinarith\n    · change Convex ℝ (L ⁻¹' Set.Ici (Real.sqrt 3))\n      exact (convex_Ici (Real.sqrt 3) : Convex ℝ (Set.Ici (Real.sqrt 3))).linear_preimage L\n  have hbound := hsub hq\n  have hpL : L p = Real.sqrt 3 := by\n    simp [L, p, param_AI]\n    ring\n  change Real.sqrt 3 ≤ L q at hbound\n  simp only [q, L, LinearMap.coe_mk, AddHom.coe_mk] at hbound hpL\n  dsimp only at hpL\n  rw [hpL] at hbound\n  linarith",
        "the_lower_endpoint_lies_on_edge_BG_and_is_not_an_interior_point",
    ),
    proof(
        "robustpa_brumo_2025_3", "inside_BGC", "negative",
        "by\n  intro h\n  let p := param_AI (1/3)\n  have hp : p ∈ interior triangle_BGC := h (1/3) (by norm_num)\n  rcases Metric.isOpen_iff.1 isOpen_interior p hp with ⟨ε, hε, hball⟩\n  let q : ℝ × ℝ := (p.1, p.2 + ε/2)\n  have hqball : q ∈ Metric.ball p ε := by\n    simp only [Metric.mem_ball, q, Prod.dist_eq, Real.dist_eq]\n    rw [max_lt_iff]\n    constructor\n    · simpa using hε\n    · rw [abs_of_nonneg (by linarith)]\n      linarith\n  have hq := interior_subset (hball hqball)\n  let L : (ℝ × ℝ) →ₗ[ℝ] ℝ := {\n    toFun := fun z => Real.sqrt 3 * z.1 - z.2\n    map_add' := by intro x y; simp; ring\n    map_smul' := by intro c x; simp; ring }\n  have hsub : triangle_BGC ⊆ {z : ℝ × ℝ | Real.sqrt 3 ≤ L z} := by\n    apply convexHull_min\n    · simp only [Set.insert_subset_iff, Set.singleton_subset_iff, Set.mem_setOf_eq]\n      simp [L, point_B, point_G, point_C]\n      all_goals have hs : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3\n      all_goals nlinarith\n    · change Convex ℝ (L ⁻¹' Set.Ici (Real.sqrt 3))\n      exact (convex_Ici (Real.sqrt 3) : Convex ℝ (Set.Ici (Real.sqrt 3))).linear_preimage L\n  have hbound := hsub hq\n  have hpL : L p = Real.sqrt 3 := by\n    simp [L, p, param_AI]\n    ring\n  change Real.sqrt 3 ≤ Real.sqrt 3 * p.1 - (p.2 + ε/2) at hbound\n  change Real.sqrt 3 * p.1 - p.2 = Real.sqrt 3 at hpL\n  linarith",
        "supporting_linear_halfspace_at_edge_BG_excludes_an_upward_ball_point",
    ),
    proof(
        "robustpa_hmmt_feb_2025_4", "pair_floor_sum", "positive",
        "by\n  have hm0i : m ≠ 0 := by omega\n  have hm0 : (m : ℝ) ≠ 0 := by exact_mod_cast hm0i\n  have hneg : (4050 : ℝ) / ((-m : ℤ) : ℝ) = -((4050 : ℝ) / (m : ℝ)) := by\n    push_cast\n    field_simp\n  rw [hneg, Int.floor_neg]\n  by_cases hd : m ∣ 4050\n  · rw [if_pos hd]\n    rcases hd with ⟨k, hk⟩\n    have hx : (4050 : ℝ) / (m : ℝ) = (k : ℝ) := by\n      rw [hk]\n      push_cast\n      field_simp\n    rw [hx]\n    simp\n  · rw [if_neg hd]\n    have hn : (4050 : ℝ) / (m : ℝ) ∉ Set.range Int.cast := by\n      rintro ⟨k, hk⟩\n      apply hd\n      refine ⟨k, ?_⟩\n      have he : (4050 : ℝ) = (m : ℝ) * (k : ℝ) := by\n        field_simp [hm0] at hk\n        nlinarith\n      exact_mod_cast he.symm\n    rw [(Int.ceil_eq_floor_add_one_iff_notMem _).mpr hn]\n    omega",
        "floor_x_plus_floor_neg_x_is_zero_for_integral_quotients_and_minus_one_otherwise",
    ),
    proof(
        "robustpa_hmmt_feb_2025_4", "pair_floor_sum", "positive",
        "by\n  have hm0i : m ≠ 0 := by omega\n  have hm0 : (m : ℝ) ≠ 0 := by exact_mod_cast hm0i\n  have hneg : (4050 : ℝ) / ((-m : ℤ) : ℝ) = -((4050 : ℝ) / (m : ℝ)) := by\n    push_cast\n    field_simp\n  rw [hneg, Int.floor_neg]\n  by_cases hd : m ∣ 4050\n  · rw [if_pos hd]\n    rcases hd with ⟨k, hk⟩\n    have hkr : (4050 : ℝ) = (m : ℝ) * (k : ℝ) := by exact_mod_cast hk\n    have hx : (4050 : ℝ) / (m : ℝ) = (k : ℝ) := by\n      rw [hkr]\n      field_simp\n    rw [hx]\n    simp\n  · rw [if_neg hd]\n    have hn : (4050 : ℝ) / (m : ℝ) ∉ Set.range Int.cast := by\n      rintro ⟨k, hk⟩\n      apply hd\n      refine ⟨k, ?_⟩\n      have he : (4050 : ℝ) = (m : ℝ) * (k : ℝ) := by\n        field_simp [hm0] at hk\n        nlinarith\n      exact_mod_cast he\n    rw [(Int.ceil_eq_floor_add_one_iff_notMem _).mpr hn]\n    omega",
        "cast_divisibility_witnesses_and_apply_the_floor_ceil_gap_criterion",
    ),
    proof(
        "robustpa_cmimc_2025_37", "trinomial_lucas_2024", "negative",
        "by\n  intro h\n  let c : Fin 7 → ℕ := ![0, 0, 1, 0, 0, 0, 0]\n  have hc := h 9 c (by norm_num [c, Fin.sum_univ_succ]) (by intro i; fin_cases i <;> norm_num [c])\n  norm_num [c, trinomial_coeff] at hc",
        "wrong_base_three_digit_order_fails_at_k_nine",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_contains_point_neg15", "positive",
        "by\n  let t : ℝ := Real.sqrt 2\n  let s1 : ℂ := -15 + 10*t\n  let s2 : ℂ := -15 - 10*t\n  have ht2 : t^2 = 2 := by norm_num [t]\n  have ht0 : 0 ≤ t := Real.sqrt_nonneg 2\n  refine ⟨s1, s2, ?_, ?_, ?_⟩\n  · apply Complex.ext <;> norm_num [p2, s1, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · apply Complex.ext <;> norm_num [p2, s2, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · refine ⟨(-15 : ℂ), 10*t, by positivity, ?_, ?_, ?_, ?_⟩\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]\n    · simp only [s1, Complex.norm_real, Complex.ofReal_neg, Complex.ofReal_ofNat, Complex.ofReal_mul, Complex.ofReal_sub, Complex.ofReal_add]\n      rw [Real.norm_eq_abs, abs_of_nonneg (by nlinarith)]\n    · simp only [s2, Complex.norm_real, Complex.ofReal_neg, Complex.ofReal_ofNat, Complex.ofReal_mul, Complex.ofReal_sub, Complex.ofReal_add]\n      rw [Real.norm_eq_abs, abs_of_nonpos (by nlinarith)]\n      ring",
        "explicit_real_roots_and_circle_center_minus_fifteen",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_contains_point_neg15", "positive",
        "by\n  let t : ℝ := Real.sqrt 2\n  let s1 : ℂ := -15 + 10*t\n  let s2 : ℂ := -15 - 10*t\n  have ht2 : t^2 = 2 := by norm_num [t]\n  have ht0 : 0 ≤ t := Real.sqrt_nonneg 2\n  refine ⟨s1, s2, ?_, ?_, ?_⟩\n  · apply Complex.ext <;> norm_num [p2, s1, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · apply Complex.ext <;> norm_num [p2, s2, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · refine ⟨(-15 : ℂ), 10*t, by positivity, ?_, ?_, ?_, ?_⟩\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, s1, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, s2, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]",
        "verify_all_four_radii_by_complex_norm_squared",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_contains_point_neg15", "positive",
        "by\n  let t : ℝ := Real.sqrt 2\n  let s1 : ℂ := -15 + 10*t\n  let s2 : ℂ := -15 - 10*t\n  have ht2 : t^2 = 2 := by norm_num [t]\n  have ht0 : 0 ≤ t := Real.sqrt_nonneg 2\n  refine ⟨s1, s2, ?_, ?_, ?_⟩\n  · apply Complex.ext <;> norm_num [p2, s1, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · apply Complex.ext <;> norm_num [p2, s2, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · refine ⟨(-15 : ℂ), 10*t, by positivity, ?_, ?_, ?_, ?_⟩\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, t]\n      rw [show (200 : ℝ) = (10 * Real.sqrt 2)^2 by nlinarith]\n      rw [Real.sqrt_sq (by positivity)]\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, s1, t]\n    · rw [Complex.norm_def]\n      norm_num [Complex.normSq_apply, s2, t]\n      rw [show ((-15 - 10 * Real.sqrt 2 + 15) * (-15 - 10 * Real.sqrt 2 + 15)) = (10 * Real.sqrt 2)^2 by ring]\n      rw [Real.sqrt_sq (by positivity)]",
        "close_positive_and_negative_real_root_norms_separately",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_contains_real_interval", "positive",
        "by\n  intro m hlo hhi hm\n  let t : ℝ := Real.sqrt (25 - m^2)\n  have hrad : 0 ≤ 25 - m^2 := by nlinarith\n  have ht2 : t^2 = 25 - m^2 := Real.sq_sqrt hrad\n  let s1 : ℂ := m + t * Complex.I\n  let s2 : ℂ := m - t * Complex.I\n  refine ⟨s1, s2, ?_, ?_, ?_⟩\n  · apply Complex.ext <;> simp [p2, s1, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · apply Complex.ext <;> simp [p2, s2, t, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith\n  · let center : ℂ := (10 / (m + 1) : ℝ)\n    let radius : ℝ := ‖(-1 + 2*Complex.I) - center‖\n    have hm1 : m + 1 ≠ 0 := by linarith\n    refine ⟨center, radius, ?_, rfl, ?_, ?_, ?_⟩\n    · rw [norm_pos_iff]\n      intro he\n      have hi := congrArg Complex.im he\n      norm_num [center] at hi\n    · rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n      norm_num [Complex.normSq_apply, center]\n    · rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n      simp only [Complex.normSq_apply, s1, center, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]\n      norm_num\n      field_simp [hm1]\n      nlinarith\n    · rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n      simp only [Complex.normSq_apply, s2, center, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]\n      norm_num\n      field_simp [hm1]\n      nlinarith",
        "construct_conjugate_roots_and_the_common_real_center_for_each_m_in_the_interval",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_contains_real_interval", "positive",
        "by\n  intro m hlo hhi hm\n  let t : ℝ := Real.sqrt (25 - m^2)\n  have hrad : 0 ≤ 25 - m^2 := by nlinarith\n  have ht2 : t^2 = 25 - m^2 := Real.sq_sqrt hrad\n  let s1 : ℂ := m + t * Complex.I\n  let s2 : ℂ := m - t * Complex.I\n  refine ⟨s1, s2, ?_, ?_, ?_⟩\n  · apply Complex.ext <;> simp [p2, s1, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n  · apply Complex.ext <;> simp [p2, s2, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n  · let center : ℂ := (10 / (m + 1) : ℝ)\n    let radius : ℝ := ‖(-1 + 2*Complex.I) - center‖\n    have hm1 : m + 1 ≠ 0 := by intro he; apply hm; linarith\n    refine ⟨center, radius, ?_, rfl, ?_, ?_, ?_⟩\n    · dsimp only [radius]\n      rw [norm_pos_iff]\n      intro he\n      have hi := congrArg Complex.im he\n      norm_num [center] at hi\n    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n      norm_num [Complex.normSq_apply, center]\n    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n      simp only [Complex.normSq_apply, s1, center, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]\n      norm_num\n      field_simp [hm1]\n      nlinarith [ht2]\n    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n      simp only [Complex.normSq_apply, s2, center, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]\n      norm_num\n      field_simp [hm1]\n      nlinarith [ht2]",
        "unfold_the_radius_once_and_use_the_conjugate_root_norm_squared_identity",
    ),
    proof(
        "robustpa_cmimc_2025_38", "U_contains_real_interval", "positive",
        "by\n  intro m hlo hhi hm\n  let t : ℝ := Real.sqrt (25 - m^2)\n  have hrad : 0 ≤ 25 - m^2 := by nlinarith\n  have ht2 : t^2 = 25 - m^2 := Real.sq_sqrt hrad\n  let s1 : ℂ := m + t * Complex.I\n  let s2 : ℂ := m - t * Complex.I\n  refine ⟨s1, s2, ?_, ?_, ?_⟩\n  · apply Complex.ext <;> simp [p2, s1, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n  · apply Complex.ext <;> simp [p2, s2, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n  · let center : ℂ := ((10 / (m + 1) : ℝ) : ℂ)\n    let radius : ℝ := ‖(-1 + 2*Complex.I) - center‖\n    have hm1 : m + 1 ≠ 0 := by intro he; apply hm; linarith\n    refine ⟨center, radius, ?_, rfl, ?_, ?_, ?_⟩\n    · dsimp only [radius]\n      exact norm_pos_iff.mpr (by\n        intro he\n        have hi := congrArg Complex.im he\n        norm_num [center] at hi)\n    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n      norm_num [Complex.normSq_apply, center]\n    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n      simp only [Complex.normSq_apply, s1, center, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]\n      norm_num\n      field_simp [hm1]\n      nlinarith [ht2]\n    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n      simp only [Complex.normSq_apply, s2, center, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]\n      norm_num\n      field_simp [hm1]\n      nlinarith [ht2]",
        "force_the_circle_center_to_be_a_real_coercion_before_norm_simplification",
    ),
    proof(
        "robustpa_hmmt_feb_2025_6", "r_mod_2017", "positive",
        "by\n  have hd : 2017 ∣ N := by\n    rw [N]\n    exact Nat.dvd_factorial (by norm_num) (by norm_num)\n  rw [r, Nat.mod_mod_of_dvd _ hd]\n  have hN : 0 < N := by simp [N]\n  have hrepr : N = (N - 1) + 1 := by omega\n  rw [hrepr, pow_succ]\n  let k := 2017 ^ (N - 1)\n  have hk : 0 < k := by positivity\n  have heq : k * 2017 - 1 = (k - 1) * 2017 + 2016 := by omega\n  change (k * 2017 - 1) % 2017 = 2016\n  rw [heq]\n  norm_num",
        "collapse_the_nested_modulo_before_factoring_the_positive_power",
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q_formula_claimed", "negative",
        "by\n  intro hall\n  rcases codex_q_has_root with ⟨z, hz⟩\n  have hq : z^6 + z^5 - 17*z^4 - 11*z^3 + 91*z^2 + 25*z - 149 = 0 := by\n    simpa [Polynomial.IsRoot, codex_q] using hz\n  let a : ℂ := z\n  let b : ℂ := a^2 - 6\n  let c : ℂ := b^2 - 6\n  have hclose : c^2 = a + 6 := by\n    dsimp [a, b, c]\n    linear_combination (z - 3) * (z + 2) * hq\n  have hnotfix : a ≠ 3 ∧ a ≠ -2 := by\n    constructor <;> intro ha <;> dsimp [a] at ha <;> subst z <;> norm_num at hq\n  have hab : a ≠ b := by\n    intro he\n    have hf : (a - 3) * (a + 2) = 0 := by dsimp [b] at he; nlinarith\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · exact hnotfix.1 (sub_eq_zero.mp h3)\n    · exact hnotfix.2 (eq_neg_of_add_eq_zero_left h2)\n  have hbc : b ≠ c := by\n    intro he\n    have hf : (b - 3) * (b + 2) = 0 := by dsimp [c] at he; nlinarith\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · have hb3 : b = 3 := sub_eq_zero.mp h3\n      have ha3 : a = 3 := by rw [← hclose]; rw [← he, hb3]; norm_num\n      exact hnotfix.1 ha3\n    · have hb2 : b = -2 := eq_neg_of_add_eq_zero_left h2\n      have ha2 : a = -2 := by rw [← hclose]; rw [← he, hb2]; norm_num\n      exact hnotfix.2 ha2\n  have hca : c ≠ a := by\n    intro he\n    have hf : (a - 3) * (a + 2) = 0 := by rw [he] at hclose; nlinarith\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · exact hnotfix.1 (sub_eq_zero.mp h3)\n    · exact hnotfix.2 (eq_neg_of_add_eq_zero_left h2)\n  have hs : system_holds a b c := by\n    refine ⟨hab, hbc, hca, ?_, ?_, hclose⟩ <;> dsimp [b, c] <;> ring\n  have hwrong := hall a b c hs\n  rcases hs with ⟨_, _, _, ha, hb, hc⟩\n  have h1 : (a+b)*(b+c)*(c+a) = 1 := by\n    have hmul : ((a-b)*(b-c)*(c-a))*((a+b)*(b+c)*(c+a)) = ((a-b)*(b-c)*(c-a)) := by\n      calc\n        _ = (b-c)*(c-a)*(a-b) := by\n          rw [show (a-b)*(a+b)=b-c by nlinarith [ha], show (b-c)*(b+c)=c-a by nlinarith [hb], show (c-a)*(c+a)=a-b by nlinarith [hc]]\n          ring\n        _ = _ := by ring\n    have hD : (a-b)*(b-c)*(c-a) ≠ 0 := mul_ne_zero (mul_ne_zero (sub_ne_zero.mpr hab) (sub_ne_zero.mpr hbc)) (sub_ne_zero.mpr hca)\n    exact mul_left_cancel₀ hD hmul\n  have hid : (a+b)*(b+c)*(c+a) = (a+b+c)*(a*b+b*c+c*a)-a*b*c := by ring\n  rw [hid, hwrong] at h1\n  norm_num at h1",
        "construct_a_genuine_complex_three_cycle_and_cancel_the_nonzero_difference_product",
        prefix_code=HMMT10_PREFIX,
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q_formula_claimed", "negative",
        "by\n  intro hall\n  rcases codex_q_has_root with ⟨z, hz⟩\n  have hq : z^6 + z^5 - 17*z^4 - 11*z^3 + 91*z^2 + 25*z - 149 = 0 := by\n    simpa [Polynomial.IsRoot, codex_q] using hz\n  let a : ℂ := z\n  let b : ℂ := a^2 - 6\n  let c : ℂ := b^2 - 6\n  have hclose : c^2 = a + 6 := by\n    dsimp [a, b, c]\n    linear_combination (z - 3) * (z + 2) * hq\n  have hnotfix : a ≠ 3 ∧ a ≠ -2 := by\n    constructor <;> intro ha <;> dsimp [a] at ha <;> subst z <;> norm_num at hq\n  have hab : a ≠ b := by\n    intro he\n    have hf : (a - 3) * (a + 2) = 0 := by\n      calc\n        _ = a^2-a-6 := by ring\n        _ = 0 := by dsimp [b] at he; rw [← he]; ring\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · exact hnotfix.1 (sub_eq_zero.mp h3)\n    · exact hnotfix.2 (eq_neg_of_add_eq_zero_left h2)\n  have hbc : b ≠ c := by\n    intro he\n    have hf : (b - 3) * (b + 2) = 0 := by\n      calc\n        _ = b^2-b-6 := by ring\n        _ = 0 := by dsimp [c] at he; rw [← he]; ring\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · have hb3 : b = 3 := sub_eq_zero.mp h3\n      have ha3 : a = 3 := by rw [← he, hb3] at hclose; norm_num at hclose ⊢; linarith\n      exact hnotfix.1 ha3\n    · have hb2 : b = -2 := eq_neg_of_add_eq_zero_left h2\n      have ha2 : a = -2 := by rw [← he, hb2] at hclose; norm_num at hclose ⊢; linarith\n      exact hnotfix.2 ha2\n  have hca : c ≠ a := by\n    intro he\n    have hf : (a - 3) * (a + 2) = 0 := by\n      calc\n        _ = a^2-a-6 := by ring\n        _ = 0 := by rw [← he, hclose]; ring\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · exact hnotfix.1 (sub_eq_zero.mp h3)\n    · exact hnotfix.2 (eq_neg_of_add_eq_zero_left h2)\n  have hs : system_holds a b c := by\n    refine ⟨hab, hbc, hca, ?_, ?_, hclose⟩ <;> dsimp [b, c] <;> ring\n  have hwrong := hall a b c hs\n  rcases hs with ⟨_, _, _, ha, hb, hc⟩\n  have hr1 : (a-b)*(a+b)=b-c := by linear_combination ha-hb\n  have hr2 : (b-c)*(b+c)=c-a := by linear_combination hb-hc\n  have hr3 : (c-a)*(c+a)=a-b := by linear_combination hc-ha\n  have h1 : (a+b)*(b+c)*(c+a) = 1 := by\n    have hmul : ((a-b)*(b-c)*(c-a))*((a+b)*(b+c)*(c+a)) = ((a-b)*(b-c)*(c-a)) := by\n      calc\n        _ = ((a-b)*(a+b))*((b-c)*(b+c))*((c-a)*(c+a)) := by ring\n        _ = (b-c)*(c-a)*(a-b) := by rw [hr1, hr2, hr3]\n        _ = _ := by ring\n    have hD : (a-b)*(b-c)*(c-a) ≠ 0 := mul_ne_zero (mul_ne_zero (sub_ne_zero.mpr hab) (sub_ne_zero.mpr hbc)) (sub_ne_zero.mpr hca)\n    apply mul_left_cancel₀ hD\n    simpa using hmul\n  have hid : (a+b)*(b+c)*(c+a) = (a+b+c)*(a*b+b*c+c*a)-a*b*c := by ring\n  rw [hid, hwrong] at h1\n  norm_num at h1",
        "use_FTA_for_a_true_period_three_point_then_derive_the_correct_minus_one_identity",
        prefix_code=HMMT10_PREFIX,
    ),
    proof(
        "robustpa_hmmt_feb_2025_10", "Q_formula_claimed", "negative",
        "by\n  intro hall\n  rcases codex_q_has_root with ⟨z, hz⟩\n  have hq : z^6 + z^5 - 17*z^4 - 11*z^3 + 91*z^2 + 25*z - 149 = 0 := by simpa [Polynomial.IsRoot, codex_q] using hz\n  let a : ℂ := z\n  let b : ℂ := a^2 - 6\n  let c : ℂ := b^2 - 6\n  have hclose : c^2 = a + 6 := by\n    dsimp [a, b, c]\n    linear_combination (z - 3) * (z + 2) * hq\n  have hnotfix : a ≠ 3 ∧ a ≠ -2 := by\n    constructor <;> intro ha <;> dsimp [a] at ha <;> subst z <;> norm_num at hq\n  have hab : a ≠ b := by\n    intro he\n    have hf : (a - 3) * (a + 2) = 0 := by\n      calc\n        _ = a^2-a-6 := by ring\n        _ = 0 := by dsimp [b] at he; linear_combination -he\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · exact hnotfix.1 (sub_eq_zero.mp h3)\n    · exact hnotfix.2 (eq_neg_of_add_eq_zero_left h2)\n  have hbc : b ≠ c := by\n    intro he\n    have hf : (b - 3) * (b + 2) = 0 := by\n      calc\n        _ = b^2-b-6 := by ring\n        _ = 0 := by dsimp [c] at he; linear_combination -he\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · have hb3 : b = 3 := sub_eq_zero.mp h3\n      have ha3 : a = 3 := by rw [← he, hb3] at hclose; linear_combination -hclose\n      exact hnotfix.1 ha3\n    · have hb2 : b = -2 := eq_neg_of_add_eq_zero_left h2\n      have ha2 : a = -2 := by rw [← he, hb2] at hclose; linear_combination -hclose\n      exact hnotfix.2 ha2\n  have hca : c ≠ a := by\n    intro he\n    rw [he] at hclose\n    have hf : (a - 3) * (a + 2) = 0 := by\n      calc\n        _ = a^2-a-6 := by ring\n        _ = 0 := by linear_combination hclose\n    rcases mul_eq_zero.mp hf with h3 | h2\n    · exact hnotfix.1 (sub_eq_zero.mp h3)\n    · exact hnotfix.2 (eq_neg_of_add_eq_zero_left h2)\n  have hs : system_holds a b c := by refine ⟨hab, hbc, hca, ?_, ?_, hclose⟩ <;> dsimp [b, c] <;> ring\n  have hwrong := hall a b c hs\n  rcases hs with ⟨_, _, _, ha, hb, hc⟩\n  have hr1 : (a-b)*(a+b)=b-c := by linear_combination ha-hb\n  have hr2 : (b-c)*(b+c)=c-a := by linear_combination hb-hc\n  have hr3 : (c-a)*(c+a)=a-b := by linear_combination hc-ha\n  have h1 : (a+b)*(b+c)*(c+a) = 1 := by\n    have hmul : ((a-b)*(b-c)*(c-a))*((a+b)*(b+c)*(c+a)) = ((a-b)*(b-c)*(c-a)) := by\n      calc\n        _ = ((a-b)*(a+b))*((b-c)*(b+c))*((c-a)*(c+a)) := by ring\n        _ = (b-c)*(c-a)*(a-b) := by rw [hr1, hr2, hr3]\n        _ = _ := by ring\n    have hD : (a-b)*(b-c)*(c-a) ≠ 0 := mul_ne_zero (mul_ne_zero (sub_ne_zero.mpr hab) (sub_ne_zero.mpr hbc)) (sub_ne_zero.mpr hca)\n    apply mul_left_cancel₀ hD\n    simpa using hmul\n  have hid : (a+b)*(b+c)*(c+a) = (a+b+c)*(a*b+b*c+c*a)-a*b*c := by ring\n  rw [hid, hwrong] at h1\n  norm_num at h1",
        "period_three_root_with_complex_linear_combination_for_all_fixed_point_exclusions",
        prefix_code=HMMT10_PREFIX,
    ),
    proof(
        "robustpa_aime_2025_11", "claimed_values_complete", "negative",
        "by\n  intro h\n  let y : ℝ := (-1 + Real.sqrt 2993) / 68\n  have hs : (Real.sqrt (2993 : ℝ))^2 = 2993 := Real.sq_sqrt (by norm_num)\n  have hp : 0 ≤ Real.sqrt (2993 : ℝ) := Real.sqrt_nonneg 2993\n  have hy0 : 0 ≤ y := by dsimp [y]; nlinarith\n  have hfloor : ⌊(34*y^2+1)/4⌋ = (5 : ℤ) := by\n    rw [Int.floor_eq_iff]\n    constructor <;> norm_num [y] <;> nlinarith\n  have hinter : is_intersection_y y := by\n    simp only [is_intersection_y, parabola_x, f]\n    rw [hfloor]\n    norm_num only [Int.cast_ofNat, mul_comm (4:ℝ) 5]\n    dsimp only [f_base]\n    have hn : ¬(-1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 1) := by dsimp [y]; intro hh; nlinarith\n    have hb : 1 ≤ 34*y^2-20 ∧ 34*y^2-20 < 3 := by constructor <;> dsimp [y] <;> nlinarith\n    rw [if_neg hn, if_pos hb]\n    dsimp [y]\n    nlinarith\n  have hm := h y hy0 hinter\n  simp only [claimed_positive_y_values, List.mem_cons, List.not_mem_nil, or_false] at hm\n  rcases hm with h0|h1|h2|h3|h4|h5|h6|h7|h8|h9|h10|h11|h12|h13|h14|h15|h16|h17\n  · dsimp [y] at h0; nlinarith\n  · dsimp [y] at h1; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 273); have hdp := Real.sqrt_nonneg 273; dsimp [y] at h2; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 545); have hdp := Real.sqrt_nonneg 545; dsimp [y] at h3; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 817); have hdp := Real.sqrt_nonneg 817; dsimp [y] at h4; nlinarith\n  · dsimp [y] at h5; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 1361); have hdp := Real.sqrt_nonneg 1361; dsimp [y] at h6; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 1633); have hdp := Real.sqrt_nonneg 1633; dsimp [y] at h7; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 1905); have hdp := Real.sqrt_nonneg 1905; dsimp [y] at h8; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2177); have hdp := Real.sqrt_nonneg 2177; dsimp [y] at h9; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2449); have hdp := Real.sqrt_nonneg 2449; dsimp [y] at h10; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2721); have hdp := Real.sqrt_nonneg 2721; dsimp [y] at h11; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 3097); have hdp := Real.sqrt_nonneg 3097; dsimp [y] at h12; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 3265); have hdp := Real.sqrt_nonneg 3265; dsimp [y] at h13; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 3537); have hdp := Real.sqrt_nonneg 3537; dsimp [y] at h14; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 3809); have hdp := Real.sqrt_nonneg 3809; dsimp [y] at h15; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 4081); have hdp := Real.sqrt_nonneg 4081; dsimp [y] at h16; nlinarith\n  · have hd := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 4353); have hdp := Real.sqrt_nonneg 4353; dsimp [y] at h17; nlinarith",
        "the_omitted_sqrt_2993_solution_satisfies_the_piecewise_equation_but_not_the_list",
    ),
]

_aime11_complete = next(
    e for e in reversed(MANUAL_PROOFS)
    if e["record_id"] == "robustpa_aime_2025_11" and e["node_name"] == "claimed_values_complete"
)
_aime11_complete_body = _aime11_complete["proof_body"]
for _idx in (3, 7, 9, 11):
    _aime11_complete_body = _aime11_complete_body.replace(
        f"dsimp [y] at h{_idx}; nlinarith",
        f"dsimp [y] at h{_idx}; norm_num at h{_idx}; nlinarith",
    )
MANUAL_PROOFS.append(proof(
    "robustpa_aime_2025_11", "claimed_values_complete", "negative",
    _aime11_complete_body,
    "normalize_the_four_opposite_sign_radical_equalities_before_nonlinear_arithmetic",
))

_aime11_complete_body2 = _aime11_complete_body
for _idx, _rad in ((3, 545), (7, 1633), (9, 2177), (11, 2721)):
    _aime11_complete_body2 = _aime11_complete_body2.replace(
        f"norm_num at h{_idx}; nlinarith",
        f"norm_num at h{_idx}; have he : Real.sqrt 2993 = Real.sqrt {_rad} + 2 := by linarith; rw [he] at hs; nlinarith",
    )
MANUAL_PROOFS.append(proof(
    "robustpa_aime_2025_11", "claimed_values_complete", "negative",
    _aime11_complete_body2,
    "substitute_each_offset_radical_equality_into_the_sqrt_2993_square",
))

MANUAL_PROOFS.append(proof(
    "robustpa_aime_2025_11", "claimed_values_complete", "negative",
    _aime11_complete_body2.replace("; rw [he] at hs; nlinarith", "; rw [he] at hs"),
    "rewriting_the_squared_sqrt_2993_fact_immediately_closes_each_offset_case",
))

_aime11_complete_body3 = _aime11_complete_body2.replace("; rw [he] at hs; nlinarith", "; rw [he] at hs")
for _idx, _rad in ((3, 545), (7, 1633), (9, 2177), (11, 2721)):
    _aime11_complete_body3 = _aime11_complete_body3.replace(
        f"; have he : Real.sqrt 2993 = Real.sqrt {_rad} + 2 := by linarith; rw [he] at hs",
        f"\n    have he : Real.sqrt 2993 = Real.sqrt {_rad} + 2 := by linarith\n    rw [he] at hs",
    )
MANUAL_PROOFS.append(proof(
    "robustpa_aime_2025_11", "claimed_values_complete", "negative",
    _aime11_complete_body3,
    "separate_the_local_radical_equality_proof_from_the_goal_closing_rewrite",
))

MANUAL_PROOFS.append(proof(
    "robustpa_aime_2025_11", "claimed_values_complete", "negative",
    _aime11_complete_body3.replace("    rw [he] at hs", "    rw [he] at hs\n    nlinarith [hs, hd]"),
    "combine_the_expanded_offset_square_with_the_other_radicand_square",
))

MANUAL_PROOFS.extend([
    proof(
        "robustpa_cmimc_2025_35", "expected_f_fx", "negative",
        "by\n  intro h\n  have h0 := h (0 : Fin n_val)\n  have hall : ∀ f : function_space, domain_val (f (f 0)) ≤ 25 := by\n    intro f\n    exact (f (f 0)).isLt\n  have hsum : (∑ f : function_space, (domain_val (f (f 0)) : ℚ)) ≤ num_functions * 25 := by\n    rw [← Finset.card_univ]\n    exact Finset.sum_le_card_nsmul _ _ _ (fun f _ => by exact_mod_cast hall f)\n  have hnum : 0 < (num_functions : ℚ) := by positivity\n  have havg : (∑ f : function_space, (domain_val (f (f 0)) : ℚ)) / num_functions ≤ 25 := by\n    rw [div_le_iff₀ hnum]\n    simpa [mul_comm] using hsum\n  rw [h0] at havg\n  norm_num at havg",
        "temporary_placeholder",
    ),
])

HMMT14_PATHS = [
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,2),(2,3),(3,2),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,2),(2,3),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,2),(3,1),(3,2),(2,3),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,2),(3,1),(3,2),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,2),(3,2),(2,3),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,2),(3,2),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(3,2),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(2,1),(2,2),(1,3),(2,3),(3,2),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(2,1),(2,2),(1,3),(2,3),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(2,1),(2,2),(2,3),(3,2),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(2,1),(2,2),(2,3),(3,3)],
[(0,0),(0,1),(0,2),(0,3),(1,2),(2,1),(2,2),(3,1),(3,2),(2,3),(3,3)],
]
HMMT14_VEC = "![" + ",".join(str(p) for p in HMMT14_PATHS) + "]"
MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "negative",
    "by\n  intro hbad\n  let paths : Fin 13 → List (ℕ × ℕ) := " + HMMT14_VEC + "\n"
    "  let f : Fin 13 → {path : List (ℕ × ℕ) // valid_path path} := fun i => ⟨paths i, by fin_cases i <;> decide⟩\n"
    "  have hinj : Function.Injective f := by\n    intro i j hij\n    fin_cases i <;> fin_cases j <;> simp_all [f, paths]\n"
    "  have hc := Nat.card_le_card_of_injective f hinj\n  rw [← path_count, hbad] at hc\n  norm_num at hc",
    "thirteen_explicit_distinct_valid_paths_contradict_the_claimed_cardinality_twelve",
))
MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "negative",
    "by\n  intro hbad\n  let paths : Fin 13 → List (ℕ × ℕ) := " + HMMT14_VEC + "\n"
    "  let f : Fin 13 → {path : List (ℕ × ℕ) // valid_path path} := fun i => ⟨paths i, by fin_cases i <;> norm_num [valid_path, is_valid_move]⟩\n"
    "  have hinj : Function.Injective f := by\n    intro i j hij\n    have hv := congrArg Subtype.val hij\n    fin_cases i <;> fin_cases j <;> simp_all [f, paths]\n"
    "  have hc := Nat.card_le_card_of_injective f hinj\n  rw [← path_count, hbad] at hc\n  norm_num at hc",
    "prove_each_explicit_path_by_normalizing_the_move_predicate",
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "negative",
    "by\n  intro hbad\n  let paths : Fin 13 → List (ℕ × ℕ) := " + HMMT14_VEC + "\n"
    "  let f : Fin 13 → {path : List (ℕ × ℕ) // valid_path path} := fun i => ⟨paths i, by\n"
    "    change valid_path (paths i)\n"
    "    fin_cases i <;> simp only [paths] <;>\n"
    "      refine ⟨by decide, by decide, by decide, by decide, ?_⟩ <;>\n"
    "      intro k hk <;> interval_cases k <;> norm_num [is_valid_move]⟩\n"
    "  have hinj : Function.Injective f := by\n"
    "    intro i j hij\n    have hv := congrArg Subtype.val hij\n"
    "    fin_cases i <;> fin_cases j <;> simp_all [f, paths]\n"
    "  have hc := Nat.card_le_card_of_injective f hinj\n"
    "  rw [← path_count, hbad] at hc\n  norm_num at hc",
    "thirteen_concrete_paths_with_each_bounded_move_index_split_explicitly",
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "negative",
    "by\n  intro hbad\n  let paths : Fin 13 → List (ℕ × ℕ) := " + HMMT14_VEC + "\n"
    "  let f : Fin 13 → {path : List (ℕ × ℕ) // valid_path path} := fun i => ⟨paths i, by\n"
    "    change valid_path (paths i)\n"
    "    fin_cases i <;> simp only [paths] <;>\n"
    "      refine ⟨by decide, by decide, by decide, by decide, ?_⟩ <;>\n"
    "      intro k hk <;> have hk' : k ≤ 10 := by omega <;>\n"
    "      interval_cases k <;> norm_num [is_valid_move] at hk ⊢⟩\n"
    "  have hp : Function.Injective paths := by decide\n"
    "  have hinj : Function.Injective f := fun i j hij => hp (congrArg Subtype.val hij)\n"
    "  have hc := Nat.card_le_card_of_injective f hinj\n"
    "  rw [← path_count, hbad] at hc\n  norm_num at hc",
    "thirteen_paths_with_an_explicit_global_move_index_bound_and_decidable_injectivity",
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "negative",
    "by\n  intro hbad\n  let paths : Fin 13 → List (ℕ × ℕ) := " + HMMT14_VEC + "\n"
    "  have hvalid : ∀ i, valid_path (paths i) := by\n"
    "    intro i\n    fin_cases i <;> refine ⟨by decide, by decide, by decide, by decide, ?_⟩ <;>\n"
    "      intro k hk <;> norm_num [paths] at hk <;>\n"
    "      interval_cases k <;> norm_num [paths, is_valid_move] at hk ⊢\n"
    "  let f : Fin 13 → {path : List (ℕ × ℕ) // valid_path path} := fun i => ⟨paths i, hvalid i⟩\n"
    "  have hp : Function.Injective paths := by decide\n"
    "  have hinj : Function.Injective f := fun i j hij => hp (congrArg Subtype.val hij)\n"
    "  have hc := Nat.card_le_card_of_injective f hinj\n"
    "  rw [← path_count, hbad] at hc\n  norm_num at hc",
    "separate_path_validity_then_reduce_each_concrete_length_before_splitting_move_indices",
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_38", "U_contains_real_interval", "positive",
    "by\n  intro m hlo hhi hm\n  let t : ℝ := Real.sqrt (25 - m^2)\n"
    "  have hrad : 0 ≤ 25 - m^2 := by nlinarith\n"
    "  have ht2 : t^2 = 25 - m^2 := Real.sq_sqrt hrad\n"
    "  let s1 : ℂ := m + t * Complex.I\n  let s2 : ℂ := m - t * Complex.I\n"
    "  refine ⟨s1, s2, ?_, ?_, ?_⟩\n"
    "  · apply Complex.ext <;> simp [p2, s1, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n"
    "  · apply Complex.ext <;> simp [p2, s2, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n"
    "  · let center : ℂ := ((10 / (m + 1) : ℝ) : ℂ)\n"
    "    let radius : ℝ := ‖(-1 + 2*Complex.I) - center‖\n"
    "    have hm1 : m + 1 ≠ 0 := by intro he; apply hm; linarith\n"
    "    have hcre : center.re = 10 / (m + 1) := by simp [center]\n"
    "    have hcim : center.im = 0 := by simp [center]\n"
    "    refine ⟨center, radius, ?_, rfl, ?_, ?_, ?_⟩\n"
    "    · dsimp only [radius]\n      apply norm_pos_iff.mpr\n      intro he\n"
    "      have hi := congrArg Complex.im he\n      simp only [Complex.sub_im, Complex.add_im, Complex.neg_im, Complex.one_im, Complex.mul_im, Complex.ofNat_re, Complex.I_im, Complex.ofNat_im, mul_one, mul_zero, add_zero] at hi\n"
    "      rw [hcim] at hi\n      norm_num at hi\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n"
    "      simp only [Complex.normSq_apply, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.neg_re, Complex.neg_im, Complex.one_re, Complex.one_im, Complex.mul_re, Complex.mul_im, Complex.ofNat_re, Complex.ofNat_im, Complex.I_re, Complex.I_im]\n"
    "      rw [hcre, hcim]\n      ring\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n"
    "      simp only [Complex.normSq_apply, s1, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.neg_re, Complex.neg_im, Complex.one_re, Complex.one_im, Complex.mul_re, Complex.mul_im, Complex.ofNat_re, Complex.ofNat_im, Complex.I_re, Complex.I_im]\n"
    "      rw [hcre, hcim]\n      field_simp [hm1]\n      nlinarith [ht2]\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n"
    "      simp only [Complex.normSq_apply, s2, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, Complex.neg_re, Complex.neg_im, Complex.one_re, Complex.one_im, Complex.mul_re, Complex.mul_im, Complex.ofNat_re, Complex.ofNat_im, Complex.I_re, Complex.I_im]\n"
    "      rw [hcre, hcim]\n      field_simp [hm1]\n      nlinarith [ht2]",
    "keep_the_real_center_opaque_and_rewrite_its_real_and_imaginary_coordinates_explicitly",
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_38", "U_contains_real_interval", "positive",
    "by\n  intro m hlo hhi hm\n  let t : ℝ := Real.sqrt (25 - m^2)\n"
    "  have hrad : 0 ≤ 25 - m^2 := by nlinarith\n"
    "  have ht2 : t^2 = 25 - m^2 := Real.sq_sqrt hrad\n"
    "  let s1 : ℂ := m + t * Complex.I\n  let s2 : ℂ := m - t * Complex.I\n"
    "  refine ⟨s1, s2, ?_, ?_, ?_⟩\n"
    "  · apply Complex.ext <;> simp [p2, s1, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n"
    "  · apply Complex.ext <;> simp [p2, s2, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n"
    "  · let center : ℂ := Complex.ofReal (10 / (m + 1))\n"
    "    let radius : ℝ := ‖(-1 + 2*Complex.I) - center‖\n"
    "    have hm1 : m + 1 ≠ 0 := by intro he; apply hm; linarith\n"
    "    refine ⟨center, radius, ?_, rfl, ?_, ?_, ?_⟩\n"
    "    · dsimp only [radius]\n      apply norm_pos_iff.mpr\n      intro he\n"
    "      have hi := congrArg Complex.im he\n      norm_num [center] at hi\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n"
    "      norm_num [Complex.normSq_apply, center]\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n"
    "      norm_num [Complex.normSq_apply, s1, center]\n      field_simp [hm1]\n      nlinarith [ht2]\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n"
    "      norm_num [Complex.normSq_apply, s2, center]\n      field_simp [hm1]\n      nlinarith [ht2]",
    "use_Complex_ofReal_explicitly_so_center_coordinates_reduce_definitionally",
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_38", "U_contains_real_interval", "positive",
    "by\n  intro m hlo hhi hm\n  let t : ℝ := Real.sqrt (25 - m^2)\n"
    "  have hrad : 0 ≤ 25 - m^2 := by nlinarith\n"
    "  have ht2 : t^2 = 25 - m^2 := Real.sq_sqrt hrad\n"
    "  let s1 : ℂ := m + t * Complex.I\n  let s2 : ℂ := m - t * Complex.I\n"
    "  refine ⟨s1, s2, ?_, ?_, ?_⟩\n"
    "  · apply Complex.ext <;> simp [p2, s1, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n"
    "  · apply Complex.ext <;> simp [p2, s2, pow_two, Complex.mul_re, Complex.mul_im] <;> change _ = _ <;> nlinarith [ht2]\n"
    "  · let center : ℂ := ⟨10 / (m + 1), 0⟩\n"
    "    let radius : ℝ := ‖(-1 + 2*Complex.I) - center‖\n"
    "    have hm1 : m + 1 ≠ 0 := by intro he; apply hm; linarith\n"
    "    refine ⟨center, radius, ?_, rfl, ?_, ?_, ?_⟩\n"
    "    · dsimp only [radius]\n      apply norm_pos_iff.mpr\n      intro he\n"
    "      have hi := congrArg Complex.im he\n      norm_num [center] at hi\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def]\n      congr 1\n"
    "      norm_num [Complex.normSq_apply, center]\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n"
    "      norm_num [Complex.normSq_apply, s1, center]\n      field_simp [hm1]\n      nlinarith [ht2]\n"
    "    · dsimp only [radius]\n      rw [Complex.norm_def, Complex.norm_def]\n      congr 1\n"
    "      norm_num [Complex.normSq_apply, s2, center]\n      field_simp [hm1]\n      nlinarith [ht2]",
    "define_the_real_center_by_its_complex_structure_fields_to_avoid_division_coercion_rewriting",
))

HMMT6_MOD_PREFIX = """
lemma codex_hmmt6_option_anchor : True := by trivial
lemma codex_mul_sub_one_mod (k p : ℕ) (hk : 0 < k) (hp : 0 < p) :
    (k * p - 1) % p = p - 1 := by
  have hkrep : k = (k - 1) + 1 := by omega
  have he : k * p - 1 = (k - 1) * p + (p - 1) := by
    nth_rw 1 [hkrep]
    rw [add_mul]
    omega
  rw [he, Nat.add_mod, Nat.mul_mod]
  simp [hp.ne', Nat.mod_eq_of_lt (by omega : p - 1 < p)]
"""

HMMT6_DIV_PREFIX = HMMT6_MOD_PREFIX + """
lemma codex_dvd_sub_one_mod (a p : ℕ) (ha : 0 < a) (hp : 0 < p) (hd : p ∣ a) :
    (a - 1) % p = p - 1 := by
  rcases hd with ⟨k, rfl⟩
  have hk : 0 < k := by nlinarith
  simpa [Nat.mul_comm] using codex_mul_sub_one_mod k p hk hp
"""

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_6", "r_mod_2017", "positive",
    "by\n  have hd : 2017 ∣ N := by\n    rw [N]\n    exact Nat.dvd_factorial (by norm_num) (by norm_num)\n"
    "  set_option maxRecDepth 100000 in\n    unfold r\n  rw [Nat.mod_mod_of_dvd _ hd]\n"
    "  have hN : 0 < N := by rw [N]; exact Nat.factorial_pos _\n"
    "  have hNs : N - 1 + 1 = N := Nat.sub_add_cancel (Nat.one_le_iff_ne_zero.mpr hN.ne')\n"
    "  nth_rw 1 [← hNs]\n  rw [pow_succ]\n"
    "  apply codex_mul_sub_one_mod\n  · exact pow_pos (by norm_num) _\n  · norm_num",
    "factor_the_power_symbolically_and_apply_a_general_multiple_minus_one_remainder_lemma",
    prefix_code=HMMT6_MOD_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_6", "r_mod_2017", "positive",
    "by\n  have hdN : 2017 ∣ N := by\n    rw [N]\n    exact Nat.dvd_factorial (by norm_num) (by norm_num)\n"
    "  set_option maxRecDepth 100000 in\n    unfold r\n  rw [Nat.mod_mod_of_dvd _ hdN]\n"
    "  have hN : 0 < N := by rw [N]; exact Nat.factorial_pos _\n"
    "  have hpow : 2017 ∣ 2017 ^ N := dvd_pow_self 2017 (Nat.ne_of_gt hN)\n"
    "  exact codex_dvd_sub_one_mod (2017 ^ N) 2017 (by positivity) (by norm_num) hpow",
    "use_divisibility_of_a_positive_power_without_expanding_the_factorial_exponent",
    prefix_code=HMMT6_DIV_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_6", "r_mod_2017", "positive",
    "by\n"
    "  have hdN : 2017 ∣ N := by\n"
    "    rw [N]\n"
    "    exact Nat.dvd_factorial (by norm_num) (by norm_num)\n"
    "  have hN : 0 < N := by\n"
    "    rw [N]\n"
    "    exact Nat.factorial_pos _\n"
    "  have hpow : 2017 ∣ 2017 ^ N := dvd_pow_self 2017 (Nat.ne_of_gt hN)\n"
    "  have hrN : r ≡ 2017 ^ N - 1 [MOD N] := by\n"
    "    rw [r]\n"
    "    exact Nat.mod_modEq _ _\n"
    "  have hrp : r ≡ 2017 ^ N - 1 [MOD 2017] := hrN.of_dvd hdN\n"
    "  have hs : 2017 ^ N - 1 ≡ 2016 [MOD 2017] := by\n"
    "    show (2017 ^ N - 1) % 2017 = 2016 % 2017\n"
    "    rw [codex_dvd_sub_one_mod (2017 ^ N) 2017 (by positivity) (by norm_num) hpow]\n"
    "    norm_num\n"
    "  exact hrp.trans hs",
    "keep_the_factorial_definition_irreducible_while_reasoning_with_modular_equivalence",
    prefix_code=HMMT6_DIV_PREFIX + "\nattribute [irreducible] N\n",
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_37", "trinomial_lucas_2024", "negative",
    "by\n  intro h\n  let c : Fin 7 → ℕ := ![0, 0, 1, 0, 0, 0, 0]\n"
    "  have hc := h 9 c (by norm_num [c, Fin.sum_univ_succ]) (by intro i; fin_cases i <;> norm_num [c])\n"
    "  have hcoef : trinomial_coeff 2024 9 % 3 = 2 := by\n"
    "    set_option maxHeartbeats 0 in\n"
    "      norm_num [trinomial_coeff, add_pow, Polynomial.coeff_sum, Polynomial.coeff_mul, Polynomial.coeff_X_pow]\n"
    "  rw [hcoef] at hc\n  norm_num [c, trinomial_coeff] at hc",
    "expand_the_fixed_degree_coefficient_with_add_pow_before_normalization",
))

CMIMC37_PREFIX = """
noncomputable def codexF (a : ℕ) : Polynomial (ZMod 3) := 1 + Polynomial.X^a + Polynomial.X^(2*a)
lemma codexF_cube (a : ℕ) : codexF a ^ 3 = codexF (3*a) := by
  simp only [codexF]
  rw [add_pow_char, add_pow_char]
  simp only [one_pow, ← pow_mul]
  congr 2 <;> simp [mul_comm, mul_left_comm, mul_assoc]

lemma codexF_sq (a : ℕ) : codexF a ^ 2 =
    1 + 2 * Polynomial.X^a + 3 * Polynomial.X^(2*a) +
      2 * Polynomial.X^(3*a) + Polynomial.X^(4*a) := by
  unfold codexF
  ring

lemma codexF_sq_coeff_below (a m : ℕ) (hm : m < a) :
    (codexF a ^ 2).coeff m = if m = 0 then 1 else 0 := by
  rw [codexF_sq]
  by_cases h0 : m = 0
  · subst m
    have ha : a ≠ 0 := Nat.ne_of_gt hm
    have ha' : 0 ≠ a := Ne.symm ha
    simp [Polynomial.coeff_one, Polynomial.coeff_X_pow, ha, ha']
  · have hma : m ≠ a := by omega
    have hm2 : m ≠ 2*a := by omega
    have hm3 : m ≠ 3*a := by omega
    have hm4 : m ≠ 4*a := by omega
    simp [Polynomial.coeff_one, Polynomial.coeff_X_pow, h0, hma, hm2, hm3, hm4]

lemma codex_coeff_mul_far (p : Polynomial (ZMod 3)) (a n : ℕ) (hn : n < a) :
    (p * codexF a ^ 2).coeff n = p.coeff n := by
  rw [Polynomial.coeff_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk]
  calc
    ∑ k ∈ Finset.range n.succ, p.coeff k * (codexF a ^ 2).coeff (n-k) =
        p.coeff n * (codexF a ^ 2).coeff 0 := by
      rw [Finset.sum_eq_single n]
      · simp
      · intro k hk hkn
        have hkle : k ≤ n := Finset.mem_range_succ_iff.mp hk
        have hsub_lt : n - k < a := lt_of_le_of_lt (Nat.sub_le n k) hn
        have hsub_ne : n - k ≠ 0 := Nat.sub_ne_zero_iff_lt.mpr (lt_of_le_of_ne hkle hkn)
        rw [codexF_sq_coeff_below a (n-k) hsub_lt]
        simp [hsub_ne]
      · simp
    _ = p.coeff n := by
      rw [codexF_sq_coeff_below a 0 (by omega)]
      simp
"""

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_37", "trinomial_lucas_2024", "negative",
    "by\n  intro h\n  let c : Fin 7 → ℕ := ![0, 0, 1, 0, 0, 0, 0]\n"
    "  have hc := h 9 c (by norm_num [c, Fin.sum_univ_succ]) (by intro i; fin_cases i <;> norm_num [c])\n"
    "  let P : Polynomial (ZMod 3) := codexF 1\n"
    "  have hdecomp : P^2024 = codexF 1^2 * codexF 3^2 * codexF 9^2 * codexF 27^2 * codexF 81^0 * codexF 243^2 * codexF 729^2 := by\n"
    "    dsimp [P]\n"
    "    rw [show 2024 = 2 + 3*674 by norm_num, pow_add, pow_mul, codexF_cube]\n"
    "    rw [show 674 = 2 + 3*224 by norm_num, pow_add, pow_mul, codexF_cube]\n"
    "    rw [show 224 = 2 + 3*74 by norm_num, pow_add, pow_mul, codexF_cube]\n"
    "    rw [show 74 = 2 + 3*24 by norm_num, pow_add, pow_mul, codexF_cube]\n"
    "    rw [show 24 = 0 + 3*8 by norm_num, pow_add, pow_mul, codexF_cube]\n"
    "    rw [show 8 = 2 + 3*2 by norm_num, pow_add, pow_mul, codexF_cube]\n"
    "    norm_num\n    ring\n"
    "  have hcoefZ : (trinomial_coeff 2024 9 : ZMod 3) = 1 := by\n"
    "    have hm : Polynomial.map (Nat.castRingHom (ZMod 3)) ((1 + Polynomial.X + Polynomial.X^2 : Polynomial ℕ)^2024) = P^2024 := by\n"
    "      simp [P, codexF]\n"
    "    have hm9 := congrArg (fun q : Polynomial (ZMod 3) => q.coeff 9) hm\n"
    "    rw [hdecomp] at hm9\n"
    "    have heval : (codexF 1^2 * codexF 3^2 * codexF 9^2 * codexF 27^2 * codexF 81^0 * codexF 243^2 * codexF 729^2).coeff 9 = 1 := by\n"
    "      norm_num\n"
    "      rw [codex_coeff_mul_far _ 729 9 (by norm_num)]\n"
    "      rw [codex_coeff_mul_far _ 243 9 (by norm_num)]\n"
    "      rw [codex_coeff_mul_far _ 27 9 (by norm_num)]\n"
    "      rw [codexF_sq, codexF_sq, codexF_sq]\n"
    "      ring_nf\n"
    "      norm_num [Polynomial.coeff_add, Polynomial.coeff_X, Polynomial.coeff_X_pow, Polynomial.coeff_one]\n"
    "      decide\n"
    "    rw [heval, Polynomial.coeff_map] at hm9\n"
    "    set_option maxRecDepth 100000 in\n"
    "      change (((1 + Polynomial.X + Polynomial.X^2 : Polynomial ℕ)^2024).coeff 9 : ZMod 3) = 1 at hm9\n"
    "    exact hm9\n"
    "  have hcoef : trinomial_coeff 2024 9 % 3 = 1 := by\n"
    "    exact (ZMod.natCast_eq_natCast_iff' (trinomial_coeff 2024 9) 1 3).mp hcoefZ\n"
    "  have hz : trinomial_coeff (![2, 2, 0, 2, 2, 2, 2] (2 : Fin 7)) (c 2) = 0 := by\n"
    "    change ((1 + Polynomial.X + Polynomial.X^2 : Polynomial ℕ)^0).coeff 1 = 0\n"
    "    simp [Polynomial.coeff_one]\n"
    "  have hp : (∏ i : Fin 7, trinomial_coeff (![2, 2, 0, 2, 2, 2, 2] i) (c i)) = 0 :=\n"
    "    Finset.prod_eq_zero (Finset.mem_univ (2 : Fin 7)) hz\n"
    "  rw [hcoef, hp] at hc\n  norm_num at hc",
    "map_to_ZMod_three_factor_by_base_three_Frobenius_and_compute_only_coefficient_nine",
    prefix_code=CMIMC37_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_6", "r_mod_2017", "positive",
    "by\n  have hdN : 2017 ∣ N := by\n    rw [N]\n    exact Nat.dvd_factorial (by norm_num) (by norm_num)\n"
    "  have hN : 0 < N := by rw [N]; exact Nat.factorial_pos _\n"
    "  have hpow : 2017 ∣ 2017 ^ N := dvd_pow_self 2017 (Nat.ne_of_gt hN)\n"
    "  have hrdef : r = (2017 ^ N - 1) % N := rfl\n"
    "  have hrN0 := Nat.mod_modEq (2017 ^ N - 1) N\n"
    "  have hrN : r ≡ 2017 ^ N - 1 [MOD N] := hrdef ▸ hrN0\n"
    "  have hrp : r ≡ 2017 ^ N - 1 [MOD 2017] := hrN.of_dvd hdN\n"
    "  have hs : 2017 ^ N - 1 ≡ 2016 [MOD 2017] := by\n"
    "    show (2017 ^ N - 1) % 2017 = 2016 % 2017\n"
    "    rw [codex_dvd_sub_one_mod (2017 ^ N) 2017 (by positivity) (by norm_num) hpow]\n    norm_num\n"
    "  exact hrp.trans hs",
    "rewrite_r_via_a_shallow_definitional_equality_then_use_modular_transitivity",
    prefix_code=HMMT6_DIV_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_4", "sum_odd_form", "positive",
    "by\n  classical\n  unfold target_sum\n"
    "  apply Finset.sum_nbij (fun j : ℤ => 2*j+1)\n"
    "  · intro j hj\n    simp only [Finset.mem_Icc, Finset.mem_filter] at hj ⊢\n"
    "    constructor <;> omega\n"
    "  · intro a ha b hb hab\n    omega\n"
    "  · intro m hm\n    simp only [Finset.mem_filter, Finset.mem_Icc] at hm\n"
    "    refine ⟨(m-1)/2, ?_, ?_⟩\n"
    "    · simp only [Finset.mem_Icc]\n      omega\n"
    "    · omega\n"
    "  · intro j hj\n    exact congrArg Int.floor (sum_rewrite j)",
    "reindex_the_finite_sum_by_the_integer_bijection_j_maps_to_two_j_plus_one",
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_4", "sum_odd_form", "positive",
    "by\n  classical\n  unfold target_sum\n"
    "  apply Finset.sum_bij (fun j _ => 2*j+1)\n"
    "  · intro j hj\n    simp only [Finset.mem_Icc, Finset.mem_filter] at hj ⊢\n"
    "    constructor <;> omega\n"
    "  · intro a ha b hb hab\n    have he : 2*a = 2*b := by linarith\n    omega\n"
    "  · intro m hm\n    simp only [Finset.mem_filter, Finset.mem_Icc] at hm\n"
    "    refine ⟨(m-1)/2, ?_, ?_⟩\n"
    "    · simp only [Finset.mem_Icc]\n      omega\n"
    "    · omega\n"
    "  · intro j hj\n    exact congrArg Int.floor (sum_rewrite j)",
    "use_the_membership_dependent_sum_bijection_and_the_explicit_inverse_integer_quotient",
))

HMMT13_K0_PREFIX = """
def codex_k0_a : Finset (Finset ℕ) :=
  {{1,2,5,6}, {3,4,7,8}, {9,10,13,14}, {11,12,15,16}}
def codex_k0_b : Finset (Finset ℕ) :=
  {{1,2,5,6}, {3,4,9,10}, {7,8,13,14}, {11,12,15,16}}

lemma codex_valid_a : is_valid_partition codex_k0_a := by
  norm_num [codex_k0_a, is_valid_partition, is_valid_box, are_neighbors] <;> decide
lemma codex_valid_b : is_valid_partition codex_k0_b := by
  norm_num [codex_k0_b, is_valid_partition, is_valid_box, are_neighbors] <;> decide

lemma codex_two_1_5 : is_two_blocks_2 {1,2,5,6} := by
  exact ⟨1, 5, by norm_num, by norm_num, by norm_num, by decide⟩
lemma codex_two_3_7 : is_two_blocks_2 {3,4,7,8} := by
  exact ⟨3, 7, by norm_num, by norm_num, by norm_num, by decide⟩
lemma codex_two_9_13 : is_two_blocks_2 {9,10,13,14} := by
  exact ⟨9, 13, by norm_num, by norm_num, by norm_num, by decide⟩
lemma codex_two_11_15 : is_two_blocks_2 {11,12,15,16} := by
  exact ⟨11, 15, by norm_num, by norm_num, by norm_num, by decide⟩
lemma codex_two_3_9 : is_two_blocks_2 {3,4,9,10} := by
  exact ⟨3, 9, by norm_num, by norm_num, by norm_num, by decide⟩
lemma codex_two_7_13 : is_two_blocks_2 {7,8,13,14} := by
  exact ⟨7, 13, by norm_num, by norm_num, by norm_num, by decide⟩

lemma codex_k0_a_prop : is_valid_partition codex_k0_a ∧
    (∀ box ∈ codex_k0_a, is_two_blocks_2 box) := by
  refine ⟨codex_valid_a, ?_⟩
  intro box hb
  simp only [codex_k0_a, Finset.mem_insert, Finset.mem_singleton] at hb
  rcases hb with h|h|h|h <;> subst box
  · exact codex_two_1_5
  · exact codex_two_3_7
  · exact codex_two_9_13
  · exact codex_two_11_15
lemma codex_k0_b_prop : is_valid_partition codex_k0_b ∧
    (∀ box ∈ codex_k0_b, is_two_blocks_2 box) := by
  refine ⟨codex_valid_b, ?_⟩
  intro box hb
  simp only [codex_k0_b, Finset.mem_insert, Finset.mem_singleton] at hb
  rcases hb with h|h|h|h <;> subst box
  · exact codex_two_1_5
  · exact codex_two_3_9
  · exact codex_two_7_13
  · exact codex_two_11_15

lemma codex_partition_boxes_bounded {boxes : Finset (Finset ℕ)}
    (h : is_valid_partition boxes) : boxes ⊆ (Finset.Icc 1 16).powerset := by
  intro box hb
  rw [Finset.mem_powerset]
  intro x hx
  have hx' : x ∈ Finset.biUnion boxes id := Finset.mem_biUnion.mpr ⟨box, hb, hx⟩
  rw [h.2.2.1] at hx'
  exact hx'
"""

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_13", "case_k_0_count", "negative",
    "by\n  intro hbad\n"
    "  let T := {boxes : Finset (Finset ℕ) | is_valid_partition boxes ∧ (∀ box ∈ boxes, is_two_blocks_2 box)}\n"
    "  let U := (Finset.Icc 1 16).powerset.powerset\n"
    "  let e : T → {boxes : Finset (Finset ℕ) // boxes ∈ U} := fun z => ⟨z, by\n"
    "    rw [Finset.mem_powerset]\n    exact codex_partition_boxes_bounded z.property.1⟩\n"
    "  letI : Finite T := Finite.of_injective e (by\n    intro x y h\n    apply Subtype.ext\n    simpa [e] using congrArg Subtype.val h)\n"
    "  let f : Fin 2 → T := ![⟨codex_k0_a, codex_k0_a_prop⟩, ⟨codex_k0_b, codex_k0_b_prop⟩]\n"
    "  have hf : Function.Injective f := by\n    intro i j h\n    have hv := congrArg Subtype.val h\n    fin_cases i <;> fin_cases j <;> simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one] at hv ⊢\n    all_goals exfalso; revert hv; decide\n"
    "  have hc := Nat.card_le_card_of_injective f hf\n"
    "  rw [show Nat.card T = count_case_k_0 by rfl, hbad] at hc\n  norm_num at hc",
    "two_explicit_distinct_all_two_block_partitions_contradict_the_claimed_unique_count",
    prefix_code=HMMT13_K0_PREFIX,
))

HMMT14_EXPLICIT_PREFIX = "\n\n".join(
    f"lemma codex_path_{i}_valid : valid_path {path} := by\n"
    "  refine ⟨by decide, by decide, by decide, by decide, ?_⟩\n"
    "  intro k hk\n"
    "  norm_num at hk\n"
    "  interval_cases k <;> norm_num [is_valid_move]"
    for i, path in enumerate(HMMT14_PATHS)
)

HMMT14_EXPLICIT_PREFIX2 = "\n\n".join(
    f"lemma codex_path2_{i}_valid : valid_path {path} := by\n"
    "  refine ⟨by decide, by decide, by decide, by decide, ?_⟩\n"
    "  intro k hk\n"
    f"  have hk_cases : " + " ∨ ".join(f"k = {j}" for j in range(len(path) - 1)) + " := by norm_num at hk; omega\n"
    "  rcases hk_cases with " + "|".join("rfl" for _ in range(len(path) - 1)) +
    " <;> norm_num [is_valid_move]"
    for i, path in enumerate(HMMT14_PATHS)
)

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "negative",
    "by\n  intro hbad\n"
    "  let T := {path : List (ℕ × ℕ) // valid_path path}\n"
    "  let f : Fin 13 → T := ![" + ",".join(
        f"⟨{path}, codex_path_{i}_valid⟩" for i, path in enumerate(HMMT14_PATHS)
    ) + "]\n"
    "  have hf : Function.Injective f := by\n"
    "    intro i j h\n    have hv := congrArg Subtype.val h\n"
    "    fin_cases i <;> fin_cases j <;> simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one] at hv ⊢\n"
    "    all_goals exfalso; revert hv; decide\n"
    "  by_cases hfin : Finite T\n"
    "  · letI : Finite T := hfin\n"
    "    have hc := Nat.card_le_card_of_injective f hf\n"
    "    rw [show Nat.card T = path_count by rfl, hbad] at hc\n    norm_num at hc\n"
    "  · letI : Infinite T := not_finite_iff_infinite.mp hfin\n"
    "    have hz : Nat.card T = 0 := Nat.card_eq_zero_of_infinite\n"
    "    rw [show Nat.card T = path_count by rfl, hbad] at hz\n    norm_num at hz",
    "thirteen_explicit_paths_and_a_finite_or_infinite_cardinality_split",
    prefix_code=HMMT14_EXPLICIT_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_14", "dfs_count_lemma", "negative",
    "by\n  intro hbad\n"
    "  let T := {path : List (ℕ × ℕ) // valid_path path}\n"
    "  let f : Fin 13 → T := ![" + ",".join(
        f"⟨{path}, codex_path2_{i}_valid⟩" for i, path in enumerate(HMMT14_PATHS)
    ) + "]\n"
    "  have hf : Function.Injective f := by\n"
    "    intro i j h\n    have hv := congrArg Subtype.val h\n"
    "    fin_cases i <;> fin_cases j <;> simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one] at hv ⊢\n"
    "    all_goals exfalso; revert hv; decide\n"
    "  by_cases hfin : Finite T\n"
    "  · letI : Finite T := hfin\n"
    "    have hc := Nat.card_le_card_of_injective f hf\n"
    "    rw [show Nat.card T = path_count by rfl, hbad] at hc\n    norm_num at hc\n"
    "  · letI : Infinite T := not_finite_iff_infinite.mp hfin\n"
    "    have hz : Nat.card T = 0 := Nat.card_eq_zero_of_infinite\n"
    "    rw [show Nat.card T = path_count by rfl, hbad] at hz\n    norm_num at hz",
    "enumerate_each_allowed_move_index_by_an_explicit_finite_disjunction",
    prefix_code=HMMT14_EXPLICIT_PREFIX2,
))

HMMT12_POS = [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1)]
from itertools import combinations
HMMT12_SETS = []
for _pair in [((0,1),(1,0)), ((1,2),(2,1))]:
    for _extra in combinations([p for p in HMMT12_POS if p not in _pair], 2):
        _s = tuple(sorted((*_pair, *_extra)))
        if _s not in HMMT12_SETS:
            HMMT12_SETS.append(_s)
HMMT12_SETS = HMMT12_SETS[:17]

def _grid_finset(points):
    return "({" + ",".join(f"(⟨{p[0]},{p[1]}⟩ : GridPos)" for p in points) + "} : Finset GridPos)"

HMMT12_PREFIX = """
lemma codex_no_path_start (removed : Set GridPos)
    (h01 : ((0,1) : GridPos) ∈ removed) (h10 : ((1,0) : GridPos) ∈ removed) :
    disconnected (remaining_pads removed) start_pos end_pos := by
  rintro ⟨path, hn, hc, hp, hr⟩
  rcases path with _ | a :: tail
  · exact hn rfl
  · have ha : a = start_pos := by simpa [connects] using hc.1
    subst a
    rcases tail with _ | b :: rest
    · norm_num [connects, start_pos, end_pos] at hc
    · have hab : adjacent start_pos b := by simpa [is_valid_path] using hp.head
      have hb := hr b (by simp)
      fin_cases b.1 <;> fin_cases b.2 <;>
        simp [adjacent, start_pos, remaining_pads] at hab hb <;> aesop

lemma codex_no_path_end (removed : Set GridPos)
    (h12 : ((1,2) : GridPos) ∈ removed) (h21 : ((2,1) : GridPos) ∈ removed) :
    disconnected (remaining_pads removed) start_pos end_pos := by
  rintro ⟨path, hn, hc, hp, hr⟩
  have hend : end_pos ∈ path := by
    have hne : path ≠ [] := hn
    have := List.getLast?_eq_getLast_of_ne_nil hne
    rw [hc.2] at this
    simpa using List.getLast_mem path hne
  have he := hr end_pos hend
  have hlen : 2 ≤ path.length := by
    by_contra h
    have : path.length = 1 := by omega
    rcases path with _ | a :: tail
    · contradiction
    · cases tail <;> simp_all [connects, start_pos, end_pos]
  obtain ⟨pre, z, hpath⟩ := List.exists_eq_append_cons_of_length_pos (by omega : 0 < path.length)
  subst path
  cases pre with
  | nil => simp_all [connects, start_pos, end_pos]
  | cons a pre' =>
    have hpair := hp
    rw [List.pairwise_append] at hpair
    have haz : adjacent (List.getLast (a :: pre') (by simp)) z := by
      simpa using hpair.2.2
    have hz : z = end_pos := by simpa [connects] using hc.2
    subst z
    let q := List.getLast (a :: pre') (by simp)
    have hq := hr q (by simp [q])
    change adjacent q end_pos at haz
    fin_cases q.1 <;> fin_cases q.2 <;>
      simp [adjacent, end_pos, remaining_pads] at haz hq <;> aesop
"""

for _i, _pts in enumerate(HMMT12_SETS):
    _fs = _grid_finset(_pts)
    _iso_start = (0,1) in _pts and (1,0) in _pts
    HMMT12_PREFIX += f"\nlemma codex_rem_{_i}_prop : has_size_four (↑{_fs} : Set GridPos) ∧ is_valid_removal (↑{_fs} : Set GridPos) := by\n"
    HMMT12_PREFIX += "  constructor\n  · simp [has_size_four]\n  · constructor\n"
    HMMT12_PREFIX += "    · intro p hp\n      simp only [Finset.coe_insert, Finset.coe_singleton, Set.mem_insert_iff, Set.mem_singleton_iff] at hp\n      rcases hp with " + "|".join("rfl" for _ in _pts) + " <;> decide\n"
    if _iso_start:
        HMMT12_PREFIX += f"    · apply codex_no_path_start (↑{_fs} : Set GridPos) <;> simp\n"
    else:
        HMMT12_PREFIX += f"    · apply codex_no_path_end (↑{_fs} : Set GridPos) <;> simp\n"

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_12",
    "robustpa_qwen3_8b_math_verify_hmmt_feb_2025_robustpa_hmmt_feb_2025_12",
    "negative",
    "by\n  intro hbad\n"
    "  let T := {removed : Set GridPos // has_size_four removed ∧ is_valid_removal removed}\n"
    "  let f : Fin 17 → T := ![" + ",".join(
        f"⟨(↑{_grid_finset(pts)} : Set GridPos), codex_rem_{i}_prop⟩"
        for i, pts in enumerate(HMMT12_SETS)
    ) + "]\n"
    "  have hf : Function.Injective f := by\n"
    "    intro i j h\n    have hv := congrArg Subtype.val h\n"
    "    fin_cases i <;> fin_cases j <;> simp only [f, Matrix.cons_val_zero, Matrix.cons_val_one] at hv ⊢\n"
    "    all_goals exfalso; revert hv; decide\n"
    "  by_cases hfin : Finite T\n"
    "  · letI : Finite T := hfin\n"
    "    have hc := Nat.card_le_card_of_injective f hf\n"
    "    rw [show Nat.card T = disconnecting_count by rfl, hbad] at hc\n    norm_num at hc\n"
    "  · letI : Infinite T := not_finite_iff_infinite.mp hfin\n"
    "    have hz : Nat.card T = 0 := Nat.card_eq_zero_of_infinite\n"
    "    rw [show Nat.card T = disconnecting_count by rfl, hbad] at hz\n    norm_num at hz",
    "seventeen_explicit_removal_sets_isolating_an_endpoint_contradict_the_claimed_count_sixteen",
    prefix_code=HMMT12_PREFIX,
))

HMMT12_PREFIX2 = """
local instance codexGridDecEq : DecidableEq GridPos := by
  unfold GridPos
  infer_instance

lemma codex_no_path_from (removed : Set GridPos) (p q : GridPos) (hpq : p ≠ q)
    (hcut : ∀ z, adjacent p z → z ∈ removed) :
    disconnected (remaining_pads removed) p q := by
  rintro ⟨path, hn, hc, hpair, hr⟩
  cases path with
  | nil => exact hn rfl
  | cons a tail =>
    have ha : a = p := by simpa [connects] using hc.1
    subst a
    cases tail with
    | nil =>
      have : p = q := by simpa [connects] using hc.2
      exact hpq this
    | cons b rest =>
      unfold is_valid_path at hpair
      cases hpair with
      | cons hall _ =>
        have hab : adjacent p b := hall b (by simp)
        have hb : b ∈ remaining_pads removed := hr b (by simp)
        exact hb (hcut b hab)

lemma codex_start_cut (removed : Set GridPos)
    (h01 : ((0,1) : GridPos) ∈ removed) (h10 : ((1,0) : GridPos) ∈ removed) :
    disconnected (remaining_pads removed) start_pos end_pos := by
  apply codex_no_path_from removed start_pos end_pos (by
    intro h
    have hx := congrArg (fun p : GridPos => p.1.val) h
    norm_num [start_pos, end_pos] at hx)
  intro z hz
  rcases z with ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> simp [adjacent, start_pos] at hz <;> aesop

lemma codex_end_cut (removed : Set GridPos)
    (h12 : ((1,2) : GridPos) ∈ removed) (h21 : ((2,1) : GridPos) ∈ removed) :
    disconnected (remaining_pads removed) start_pos end_pos := by
  intro hpath
  rcases hpath with ⟨path, hn, hc, hp, hr⟩
  apply codex_no_path_from removed end_pos start_pos (by
    intro h
    have hx := congrArg (fun p : GridPos => p.1.val) h
    norm_num [start_pos, end_pos] at hx)
  · intro z hz
    rcases z with ⟨i,j⟩
    fin_cases i <;> fin_cases j <;> simp [adjacent, end_pos] at hz <;> aesop
  · refine ⟨path.reverse, by simpa, ?_, ?_, ?_⟩
    · simpa [connects] using And.intro hc.2 hc.1
    · unfold is_valid_path at hp ⊢
      have hrev := hp.reverse
      exact hrev.imp (by
        intro x y hxy
        simp only [adjacent] at hxy ⊢
        aesop)
    · intro pos hpos
      exact hr pos (by simpa using hpos)
"""
for _i, _pts in enumerate(HMMT12_SETS):
    _fs = _grid_finset(_pts)
    _iso_start = (0,1) in _pts and (1,0) in _pts
    HMMT12_PREFIX2 += f"\nlemma codex_rem2_{_i}_prop : has_size_four (↑{_fs} : Set GridPos) ∧ is_valid_removal (↑{_fs} : Set GridPos) := by\n"
    HMMT12_PREFIX2 += "  constructor\n  · constructor\n    · exact Set.toFinite _\n    · rw [Set.ncard_coe_finset]\n      decide\n  · constructor\n"
    HMMT12_PREFIX2 += f"    · intro p hp\n      change p ∈ {_fs} at hp\n"
    for _p in _pts[:-1]:
        HMMT12_PREFIX2 += "      rcases Finset.mem_insert.mp hp with h | hp\n      · subst p\n        constructor <;> intro h <;> have hx := congrArg (fun q : GridPos => (q.1.val, q.2.val)) h <;> norm_num [start_pos, end_pos] at hx\n"
    HMMT12_PREFIX2 += "      have h := Finset.mem_singleton.mp hp\n      subst p\n      constructor <;> intro h <;> have hx := congrArg (fun q : GridPos => (q.1.val, q.2.val)) h <;> norm_num [start_pos, end_pos] at hx\n"
    if _iso_start:
        HMMT12_PREFIX2 += f"    · apply codex_start_cut (↑{_fs} : Set GridPos) <;> change _ ∈ {_fs} <;> decide\n"
    else:
        HMMT12_PREFIX2 += f"    · apply codex_end_cut (↑{_fs} : Set GridPos) <;> change _ ∈ {_fs} <;> decide\n"

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_12",
    "robustpa_qwen3_8b_math_verify_hmmt_feb_2025_robustpa_hmmt_feb_2025_12",
    "negative",
    "by\n  intro hbad\n"
    "  let T := {removed : Set GridPos // has_size_four removed ∧ is_valid_removal removed}\n"
    "  let fs : Fin 17 → Finset GridPos := ![" + ",".join(
        _grid_finset(pts) for pts in HMMT12_SETS
    ) + "]\n"
    "  have hfs : Function.Injective fs := by decide\n"
    "  let f : Fin 17 → T := fun i => ⟨(↑(fs i) : Set GridPos), by\n"
    "    fin_cases i\n" + "".join(
        f"    · exact codex_rem2_{i}_prop\n" for i in range(17)
    ) + "  ⟩\n"
    "  have hf : Function.Injective f := by\n"
    "    intro i j h\n    apply hfs\n    apply Finset.coe_injective\n"
    "    exact congrArg Subtype.val h\n"
    "  by_cases hfin : Finite T\n"
    "  · letI : Finite T := hfin\n"
    "    have hc := Nat.card_le_card_of_injective f hf\n"
    "    rw [show Nat.card T = disconnecting_count by rfl, hbad] at hc\n    norm_num at hc\n"
    "  · letI : Infinite T := not_finite_iff_infinite.mp hfin\n"
    "    have hz : Nat.card T = 0 := Nat.card_eq_zero_of_infinite\n"
    "    rw [show Nat.card T = disconnecting_count by rfl, hbad] at hz\n    norm_num at hz",
    "endpoint_cut_lemma_plus_seventeen_distinct_concrete_removal_sets",
    prefix_code=HMMT12_PREFIX2,
))

def _matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for idx, b in enumerate(items[1:]):
        for rest in _matchings(items[1:idx+1] + items[idx+2:]):
            yield [(a,b), *rest]

def _hmmt13_box(pair):
    a,b = pair
    return (2*a+1,2*a+2,2*b+1,2*b+2)

def _nat_finset(vals):
    return "{" + ",".join(map(str, vals)) + "}"

def _partition(matching):
    return "{" + ",".join(_nat_finset(_hmmt13_box(p)) for p in matching) + "}"

HMMT13_MATCH = {
    k: [m for m in _matchings(list(range(8))) if sum(b == a+1 for a,b in m) == k]
    for k in range(5)
}

HMMT13_FALSE_PREFIX = ""
for _k, _take in [(2,16),(1,5)]:
    for _idx, _m in enumerate(HMMT13_MATCH[_k][:_take]):
        _boxes = _partition(_m)
        _s = "{" + ",".join(_nat_finset(_hmmt13_box(p)) for p in _m if p[1] == p[0]+1) + "}"
        HMMT13_FALSE_PREFIX += f"\nlemma codex_k{_k}_{_idx}_prop : is_valid_partition {_boxes} ∧\n"
        HMMT13_FALSE_PREFIX += f"    (∃ s : Finset (Finset ℕ), s ⊆ {_boxes} ∧ s.card = {_k} ∧\n"
        HMMT13_FALSE_PREFIX += "      (∀ box ∈ s, is_contiguous_block_4 box) ∧\n"
        HMMT13_FALSE_PREFIX += f"      (∀ box ∈ {_boxes} \\ s, is_two_blocks_2 box)) := by\n"
        HMMT13_FALSE_PREFIX += f"  refine ⟨by norm_num [is_valid_partition, is_valid_box, are_neighbors] <;> decide, ⟨{_s}, by decide, by decide, ?_, ?_⟩⟩\n"
        HMMT13_FALSE_PREFIX += "  · intro box hb\n    simp only [Finset.mem_insert, Finset.mem_singleton] at hb\n    rcases hb with " + "|".join("rfl" for _ in range(_k)) + "\n"
        for _p in [p for p in _m if p[1] == p[0]+1]:
            a,b=_p; n=2*a+1
            HMMT13_FALSE_PREFIX += f"    · exact ⟨{n}, by norm_num, by norm_num, by decide⟩\n"
        HMMT13_FALSE_PREFIX += "  · intro box hb\n    simp only [Finset.mem_sdiff, Finset.mem_insert, Finset.mem_singleton] at hb\n"
        HMMT13_FALSE_PREFIX += "    rcases hb.1 with " + "|".join("h"+str(i) for i in range(4)) + " <;> try {exfalso; apply hb.2; simp_all}\n"
        _non = [p for p in _m if p[1] != p[0]+1]
        for _p in _non:
            a,b=_p; aa=2*a+1; bb=2*b+1
            HMMT13_FALSE_PREFIX += f"    · subst box\n      exact ⟨{aa}, {bb}, by norm_num, by norm_num, by norm_num, by decide⟩\n"

def _hmmt13_false_entry(k, take, claimed):
    name = f"count_case_k_{k}"
    props = [f"codex_k{k}_{i}_prop" for i in range(take)]
    evec = ",".join(_partition(m) for m in HMMT13_MATCH[k][:take])
    body = (
        f"by\n  intro hbad\n  let T := {{boxes : Finset (Finset ℕ) // is_valid_partition boxes ∧ "
        f"(∃ s : Finset (Finset ℕ), s ⊆ boxes ∧ s.card = {k} ∧ (∀ box ∈ s, is_contiguous_block_4 box) ∧ (∀ box ∈ boxes \\ s, is_two_blocks_2 box))}}\n"
        f"  let e : Fin {take} → Finset (Finset ℕ) := ![{evec}]\n"
        "  have he : Function.Injective e := by decide\n"
        f"  let f : Fin {take} → T := fun i => ⟨e i, by\n    fin_cases i\n" +
        "".join(f"    · exact {p}\n" for p in props) + "  ⟩\n"
        "  have hf : Function.Injective f := by\n    intro i j h\n    apply he\n"
        "    exact congrArg Subtype.val h\n"
        "  by_cases hfin : Finite T\n"
        "  · letI : Finite T := hfin\n    have hc := Nat.card_le_card_of_injective f hf\n"
        f"    rw [show Nat.card T = {name} by rfl, hbad] at hc\n    norm_num at hc\n"
        "  · letI : Infinite T := not_finite_iff_infinite.mp hfin\n    have hz : Nat.card T = 0 := Nat.card_eq_zero_of_infinite\n"
        f"    rw [show Nat.card T = {name} by rfl, hbad] at hz\n    norm_num at hz"
    )
    marker = "\nlemma codex_k1_0_prop"
    if k == 2:
        selected_prefix = HMMT13_FALSE_PREFIX.split(marker, 1)[0]
    else:
        selected_prefix = marker + HMMT13_FALSE_PREFIX.split(marker, 1)[1]
    selected_prefix = (
        "lemma codex_hmmt13_option_anchor : True := by trivial\n"
        "set_option maxHeartbeats 0\n" + selected_prefix
    )
    return proof("robustpa_hmmt_feb_2025_13", f"case_k_{k}_count", "negative", body,
        f"explicitly_embed_{take}_case_k_{k}_partitions_against_claimed_{claimed}", prefix_code=selected_prefix)

MANUAL_PROOFS.extend([_hmmt13_false_entry(2,16,15), _hmmt13_false_entry(1,5,4)])

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_34", "max_area_bound", "positive",
    "by\n  intro v hv\n  rcases hv with ⟨hrange, hinj, hsimple⟩\n"
    "  have hi (i : Fin 6) : v i ∈ hexagon_points := by\n"
    "    rw [← hrange]\n    exact Set.mem_range_self i\n"
    "  have h0 := hi 0\n  have h1 := hi 1\n  have h2 := hi 2\n"
    "  have h3 := hi 3\n  have h4 := hi 4\n  have h5 := hi 5\n"
    "  have h01 : v 0 ≠ v 1 := hinj.ne (by decide)\n"
    "  have h02 : v 0 ≠ v 2 := hinj.ne (by decide)\n  have h12 : v 1 ≠ v 2 := hinj.ne (by decide)\n"
    "  have h03 : v 0 ≠ v 3 := hinj.ne (by decide)\n  have h13 : v 1 ≠ v 3 := hinj.ne (by decide)\n  have h23 : v 2 ≠ v 3 := hinj.ne (by decide)\n"
    "  have h04 : v 0 ≠ v 4 := hinj.ne (by decide)\n  have h14 : v 1 ≠ v 4 := hinj.ne (by decide)\n  have h24 : v 2 ≠ v 4 := hinj.ne (by decide)\n  have h34 : v 3 ≠ v 4 := hinj.ne (by decide)\n"
    "  have h05 : v 0 ≠ v 5 := hinj.ne (by decide)\n  have h15 : v 1 ≠ v 5 := hinj.ne (by decide)\n  have h25 : v 2 ≠ v 5 := hinj.ne (by decide)\n  have h35 : v 3 ≠ v 5 := hinj.ne (by decide)\n  have h45 : v 4 ≠ v 5 := hinj.ne (by decide)\n"
    "  simp only [hexagon_points, Set.mem_insert_iff, Set.mem_singleton_iff] at h0 h1 h2 h3 h4 h5\n"
    "  rcases h0 with h0|h0|h0|h0|h0|h0 <;> rcases h1 with h1|h1|h1|h1|h1|h1\n"
    "  all_goals try {exact (h01 (h0.trans h1.symm)).elim}\n"
    "  all_goals rcases h2 with h2|h2|h2|h2|h2|h2\n"
    "  all_goals try {exact (h02 (h0.trans h2.symm)).elim}; try {exact (h12 (h1.trans h2.symm)).elim}\n"
    "  all_goals rcases h3 with h3|h3|h3|h3|h3|h3\n"
    "  all_goals try {exact (h03 (h0.trans h3.symm)).elim}; try {exact (h13 (h1.trans h3.symm)).elim}; try {exact (h23 (h2.trans h3.symm)).elim}\n"
    "  all_goals rcases h4 with h4|h4|h4|h4|h4|h4\n"
    "  all_goals try {exact (h04 (h0.trans h4.symm)).elim}; try {exact (h14 (h1.trans h4.symm)).elim}; try {exact (h24 (h2.trans h4.symm)).elim}; try {exact (h34 (h3.trans h4.symm)).elim}\n"
    "  all_goals rcases h5 with h5|h5|h5|h5|h5|h5\n"
    "  all_goals try {exact (h05 (h0.trans h5.symm)).elim}; try {exact (h15 (h1.trans h5.symm)).elim}; try {exact (h25 (h2.trans h5.symm)).elim}; try {exact (h35 (h3.trans h5.symm)).elim}; try {exact (h45 (h4.trans h5.symm)).elim}\n"
    "  all_goals simp only [shoelace_area, next_index, Fin.sum_univ_succ, h0, h1, h2, h3, h4, h5]\n"
    "  all_goals norm_num [abs_of_nonneg, abs_of_nonpos]",
    "enumerate_the_six_given_vertices_and_normalize_the_shoelace_area_for_each_injective_ordering",
    prefix_code="lemma codex_cmimc34_option_anchor : True := by trivial\nset_option maxHeartbeats 0",
))

CMIMC35_SUM_PREFIX = """
lemma codex_sum_eval {alpha beta : Type} [Fintype alpha] [Fintype beta] [DecidableEq alpha]
    (i : alpha) (w : beta → ℚ) :
    (∑ f : alpha → beta, w (f i)) =
      Fintype.card ({j : alpha // j ≠ i} → beta) * ∑ y : beta, w y := by
  let e := Equiv.funSplitAt i beta
  calc
    _ = ∑ f : alpha → beta,
        (fun p : beta × ({j : alpha // j ≠ i} → beta) => w p.1) (e f) := by simp [e]
    _ = ∑ p : beta × ({j : alpha // j ≠ i} → beta), w p.1 := by
      simpa using e.sum_comp
        (fun p : beta × ({j : alpha // j ≠ i} → beta) => w p.1)
    _ = _ := by
      rw [Fintype.sum_prod_type]
      simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Finset.mul_sum]
"""

CMIMC35_NESTED_PREFIX = CMIMC35_SUM_PREFIX + """
lemma codex_nested_sum_25 (x : Fin n_val) (w : Fin n_val → ℚ) :
    (∑ f : function_space, w (f (f x))) =
      25^24 * w x + 24 * (25^23 * ∑ z : Fin n_val, w z) := by
  let e := Equiv.funSplitAt x (Fin n_val)
  let W := fun p : Fin n_val × ({j : Fin n_val // j ≠ x} → Fin n_val) =>
    if h : p.1 = x then w x else w (p.2 ⟨p.1, h⟩)
  calc
    _ = ∑ f : function_space, W (e f) := by
      apply Fintype.sum_congr
      intro f
      simp only [W, e, Equiv.funSplitAt_apply]
      by_cases h : f x = x <;> simp [h]
    _ = ∑ p : Fin n_val × ({j : Fin n_val // j ≠ x} → Fin n_val), W p :=
      e.sum_comp W
    _ = ∑ y : Fin n_val,
        if y = x then (25^24 : ℚ) * w x else 25^23 * ∑ z : Fin n_val, w z := by
      rw [Fintype.sum_prod_type]
      apply Fintype.sum_congr
      intro y
      by_cases hy : y = x
      · subst y
        simp only [W, ↓reduceDIte]
        rw [Finset.sum_const, Finset.card_univ]
        have hc : Fintype.card ({j : Fin n_val // j ≠ x} → Fin n_val) = 25^24 := by
          rw [Fintype.card_fun]
          norm_num [n_val, Fintype.card_subtype_compl]
        rw [hc]
        norm_num
      · simp only [W, hy, ↓reduceDIte]
        rw [codex_sum_eval (alpha := {j : Fin n_val // j ≠ x})
          (beta := Fin n_val) ⟨y, hy⟩ w]
        have hc : Fintype.card
            ({j : {j : Fin n_val // j ≠ x} // j ≠ ⟨y, hy⟩} → Fin n_val) = 25^23 := by
          rw [Fintype.card_fun]
          norm_num [n_val, Fintype.card_subtype_compl]
        rw [hc]
        norm_num [hy, pow_succ]
    _ = _ := by
      rw [show (∑ y : Fin n_val,
          if y = x then (25^24 : ℚ) * w x else 25^23 * ∑ z : Fin n_val, w z) =
        (∑ _y : Fin n_val, (25^23 * ∑ z : Fin n_val, w z : ℚ)) +
          ∑ y : Fin n_val,
            if y = x then (25^24 * w x - 25^23 * ∑ z : Fin n_val, w z : ℚ) else 0 by
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro y hy
        split_ifs <;> ring]
      rw [Fintype.sum_ite_eq']
      norm_num [n_val]
      ring

lemma codex_sum_fin25_val_sq :
    (∑ z : Fin n_val, (((z : ℕ) : ℚ)^2)) = 4900 := by
  have hn : (∑ z : Fin 25, (z.val^2 : ℕ)) = 4900 := by
    norm_num [Fin.sum_univ_succ]
  exact_mod_cast hn

lemma codex_sum_fin25_center_sq :
    (∑ z : Fin n_val, ((((z : ℕ) : ℚ) + 1 - 13)^2)) = 1300 := by
  have hn : (∑ z : Fin 25, (((z.val : ℤ) + 1 - 13)^2)) = 1300 := by
    norm_num [Fin.sum_univ_succ]
  exact_mod_cast hn

lemma codex_double_nat_sub_sq :
    (∑ x : Fin n_val, ∑ z : Fin n_val,
      (((domain_val z - domain_val x)^2 : ℕ) : ℚ)) = 32500 := by
  have hn : (∑ x : Fin 25, ∑ z : Fin 25,
      (((z.val + 1) - (x.val + 1))^2 : ℕ)) = 32500 := by
    norm_num [Fin.sum_univ_succ]
  exact_mod_cast hn

lemma codex_double_int_sub_sq :
    (∑ x : Fin n_val, ∑ z : Fin n_val,
      ((domain_val z : ℚ) - (domain_val x : ℚ))^2) = 65000 := by
  have hn : (∑ x : Fin 25, ∑ z : Fin 25,
      (((z.val + 1 : ℤ) - (x.val + 1 : ℤ))^2)) = 65000 := by
    norm_num [Fin.sum_univ_succ]
  exact_mod_cast hn
"""

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_35", "expected_f_fx", "negative",
    "by\n  intro hall\n"
    "  let x : Fin n_val := ⟨0, by norm_num [n_val]⟩\n"
    "  let e := Equiv.funSplitAt x (Fin n_val)\n"
    "  let W := fun p : Fin n_val × ({j : Fin n_val // j ≠ x} → Fin n_val) =>\n"
    "    if h : p.1 = x then (domain_val x : ℚ) else (domain_val (p.2 ⟨p.1, h⟩) : ℚ)\n"
    "  have hsum : (∑ f : function_space, (domain_val (f (f x)) : ℚ)) =\n"
    "      25^24 + 24 * (25^23 * 325) := by\n"
    "    calc\n"
    "      _ = ∑ f : function_space, W (e f) := by\n"
    "        apply Fintype.sum_congr\n        intro f\n"
    "        simp only [W, e, Equiv.funSplitAt_apply]\n"
    "        by_cases h : f x = x <;> simp [h]\n"
    "      _ = ∑ p : Fin n_val × ({j : Fin n_val // j ≠ x} → Fin n_val), W p := e.sum_comp W\n"
    "      _ = ∑ y : Fin n_val, if y = x then (25^24 : ℚ) else 25^23 * 325 := by\n"
    "        rw [Fintype.sum_prod_type]\n        apply Fintype.sum_congr\n        intro y\n"
    "        by_cases hy : y = x\n"
    "        · subst y\n          simp only [W, ↓reduceDIte]\n"
    "          rw [Finset.sum_const, Finset.card_univ]\n"
    "          have hc : Fintype.card ({j : Fin n_val // j ≠ x} → Fin n_val) = 25^24 := by\n"
    "            rw [Fintype.card_fun]\n            norm_num [n_val, Fintype.card_subtype_compl]\n"
    "          rw [hc]\n          norm_num [domain_val, x]\n"
    "        · simp only [W, hy, ↓reduceDIte]\n"
    "          have heval := codex_sum_eval (alpha := {j : Fin n_val // j ≠ x})\n"
    "            (beta := Fin n_val) ⟨y, hy⟩ (fun z => (domain_val z : ℚ))\n"
    "          rw [heval]\n"
    "          have hc : Fintype.card ({j : {j : Fin n_val // j ≠ x} // j ≠ ⟨y, hy⟩} → Fin n_val) = 25^23 := by\n"
    "            rw [Fintype.card_fun]\n            norm_num [n_val, Fintype.card_subtype_compl]\n"
    "          rw [hc]\n"
    "          have hs : (∑ z : Fin n_val, (domain_val z : ℚ)) = 325 := by\n"
    "            have hn : (∑ z : Fin 25, (z.val + 1 : ℕ)) = 325 := by\n"
    "              norm_num [Fin.sum_univ_succ]\n"
    "            exact_mod_cast hn\n"
    "          rw [hs]\n          norm_num [hy, pow_succ]\n"
    "      _ = _ := by\n"
    "        rw [show (∑ y : Fin n_val, if y = x then (25^24 : ℚ) else 25^23*325) =\n"
    "          (∑ y : Fin n_val, (25^23*325 : ℚ)) +\n"
    "            ∑ y : Fin n_val, if y = x then (25^24 - 25^23*325 : ℚ) else 0 by\n"
    "              rw [← Finset.sum_add_distrib]\n              apply Finset.sum_congr rfl\n"
    "              intro y hy\n              split_ifs <;> ring]\n"
    "        rw [Fintype.sum_ite_eq']\n        norm_num [n_val]\n"
    "  have hx := hall x\n  rw [hsum] at hx\n"
    "  norm_num [num_functions, n_val] at hx",
    "split_the_function_space_at_x_and_at_f_x_then_count_the_two_cases_exactly",
    prefix_code=CMIMC35_SUM_PREFIX,
))

# The same exact function-space split also computes the second moment.  At
# x = 1 the exceptional fixed-point branch contributes 1, while every other
# first image leaves one constrained and 23 free coordinates.
_cmimc35_fx_sq_body = MANUAL_PROOFS[-1]["proof_body"]
_cmimc35_fx_sq_body = _cmimc35_fx_sq_body.replace(
    "(domain_val (f (f x)) : ℚ)", "(domain_val (f (f x)) : ℚ)^2"
).replace(
    "(domain_val x : ℚ)", "(domain_val x : ℚ)^2"
).replace(
    "(domain_val (p.2 ⟨p.1, h⟩) : ℚ)",
    "(domain_val (p.2 ⟨p.1, h⟩) : ℚ)^2",
).replace(
    "(fun z => (domain_val z : ℚ))", "(fun z => (domain_val z : ℚ)^2)"
).replace(
    "(∑ z : Fin n_val, (domain_val z : ℚ)) = 325",
    "(∑ z : Fin n_val, (domain_val z : ℚ)^2) = 5525",
).replace(
    "(∑ z : Fin 25, (z.val + 1 : ℕ)) = 325",
    "(∑ z : Fin 25, ((z.val + 1)^2 : ℕ)) = 5525",
).replace("325", "5525")
MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_35", "expected_f_fx_sq", "negative",
    _cmimc35_fx_sq_body,
    "split_the_function_space_and_compute_the_second_moment_exactly",
    prefix_code=CMIMC35_SUM_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_35", "symmetry_of_terms", "negative",
    "by\n  intro hall\n"
    "  let x0 : Fin n_val := ⟨0, by norm_num [n_val]⟩\n"
    "  let x12 : Fin n_val := ⟨12, by norm_num [n_val]⟩\n"
    "  have h := hall x0 x12\n"
    "  have h0 := codex_nested_sum_25 x0 (fun z =>\n"
    "    ((domain_val z : ℚ) - (domain_val x0 : ℚ))^2)\n"
    "  have h12 := codex_nested_sum_25 x12 (fun z =>\n"
    "    ((domain_val z : ℚ) - (domain_val x12 : ℚ))^2)\n"
    "  rw [h0, h12] at h\n"
    "  norm_num [domain_val, x0, x12, num_functions] at h\n"
    "  rw [codex_sum_fin25_val_sq, codex_sum_fin25_center_sq] at h\n"
    "  norm_num at h",
    "compare_the_exact_second_moments_at_one_and_thirteen",
    prefix_code=CMIMC35_NESTED_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_35", "linearity_of_expectation", "negative",
    "by\n  intro hbad\n"
    "  have hleft : (∑ f : function_space, (sum_expr f : ℚ)) =\n"
    "      24 * 25^23 * 32500 := by\n"
    "    calc\n"
    "      _ = ∑ f : function_space, ∑ x : Fin n_val,\n"
    "          ((((domain_val (f (f x)) - domain_val x)^2 : ℕ) : ℚ)) := by\n"
    "        apply Fintype.sum_congr\n        intro f\n"
    "        simp only [sum_expr, Nat.cast_sum, Nat.cast_pow]\n"
    "      _ = ∑ x : Fin n_val, ∑ f : function_space,\n"
    "          ((((domain_val (f (f x)) - domain_val x)^2 : ℕ) : ℚ)) :=\n"
    "        Finset.sum_comm\n"
    "      _ = ∑ x : Fin n_val, 24 * 25^23 *\n"
    "          ∑ z : Fin n_val, ((((domain_val z - domain_val x)^2 : ℕ) : ℚ)) := by\n"
    "        apply Fintype.sum_congr\n        intro x\n"
    "        rw [codex_nested_sum_25 x (fun z =>\n"
    "          ((((domain_val z - domain_val x)^2 : ℕ) : ℚ)))]\n"
    "        simp\n        ring\n"
    "      _ = _ := by\n"
    "        rw [← Finset.mul_sum, codex_double_nat_sub_sq]\n"
    "  have hright_num : (∑ x : Fin n_val, ∑ f : function_space,\n"
    "      ((domain_val (f (f x)) : ℚ) - (domain_val x : ℚ))^2) =\n"
    "      24 * 25^23 * 65000 := by\n"
    "    calc\n"
    "      _ = ∑ x : Fin n_val, 24 * 25^23 *\n"
    "          ∑ z : Fin n_val, ((domain_val z : ℚ) - (domain_val x : ℚ))^2 := by\n"
    "        apply Fintype.sum_congr\n        intro x\n"
    "        rw [codex_nested_sum_25 x (fun z =>\n"
    "          ((domain_val z : ℚ) - (domain_val x : ℚ))^2)]\n"
    "        simp\n        ring\n"
    "      _ = _ := by\n"
    "        rw [← Finset.mul_sum, codex_double_int_sub_sq]\n"
    "  rw [hleft] at hbad\n"
    "  rw [← Finset.sum_div, hright_num] at hbad\n"
    "  norm_num [num_functions, n_val, pow_succ] at hbad",
    "compare_truncated_natural_subtraction_with_the_signed_square_sum",
    prefix_code=CMIMC35_NESTED_PREFIX,
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_37", "trinomial_lucas_2024", "negative",
    "by\n  intro h\n  let c : Fin 7 → ℕ := ![0, 0, 1, 0, 0, 0, 0]\n"
    "  have hc := h 9 c (by norm_num [c, Fin.sum_univ_succ]) (by intro i; fin_cases i <;> norm_num [c])\n"
    "  set_option maxHeartbeats 0 in\n    norm_num [c, trinomial_coeff] at hc",
    "kernel_compute_the_degree_nine_coefficient_with_unbounded_heartbeats",
))

MANUAL_PROOFS.append(proof(
    "robustpa_cmimc_2025_37", "trinomial_lucas_2024", "negative",
    "by\n  intro h\n  let c : Fin 7 → ℕ := ![0, 0, 1, 0, 0, 0, 0]\n"
    "  have hc := h 9 c (by norm_num [c, Fin.sum_univ_succ]) (by intro i; fin_cases i <;> norm_num [c])\n"
    "  have hcoef : trinomial_coeff 2024 9 % 3 = 2 := by\n"
    "    set_option maxRecDepth 100000 in\n    set_option maxHeartbeats 0 in\n      decide\n"
    "  rw [hcoef] at hc\n  norm_num [c, trinomial_coeff] at hc",
    "kernel_decide_the_single_computable_coefficient_before_simplifying_the_wrong_digit_product",
))

MANUAL_PROOFS.append(proof(
    "robustpa_hmmt_feb_2025_6", "r_mod_2017", "positive",
    "by\n  have hdN : 2017 ∣ N := by\n    rw [N]\n    exact Nat.dvd_factorial (by norm_num) (by norm_num)\n"
    "  have hN : 0 < N := by rw [N]; exact Nat.factorial_pos _\n"
    "  have hpow : 2017 ∣ 2017 ^ N := dvd_pow_self 2017 (Nat.ne_of_gt hN)\n"
    "  have hrN : r ≡ 2017 ^ N - 1 [MOD N] := by\n    rw [r]\n    exact Nat.mod_modEq _ _\n"
    "  have hrp : r ≡ 2017 ^ N - 1 [MOD 2017] := hrN.of_dvd hdN\n"
    "  have hs : 2017 ^ N - 1 ≡ 2016 [MOD 2017] := by\n"
    "    show (2017 ^ N - 1) % 2017 = 2016 % 2017\n"
    "    rw [codex_dvd_sub_one_mod (2017 ^ N) 2017 (by positivity) (by norm_num) hpow]\n    norm_num\n"
    "  have := hrp.trans hs\n  exact this",
    "route_the_remainder_through_Nat_ModEq_without_reducing_the_factorial_or_large_power",
    prefix_code=HMMT6_DIV_PREFIX,
))
