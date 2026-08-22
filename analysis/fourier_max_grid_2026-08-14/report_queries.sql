-- dataset: overview
SELECT
  COUNT(*) AS selected_runs,
  SUM(completed) AS completed_runs,
  SUM(1 - completed) AS checkpoint_runs,
  SUM(CASE WHEN selection <> 'no_explicit_geometry' THEN 1 ELSE 0 END) AS explicit_geometry_runs,
  SUM(
    CASE
      WHEN selection = 'no_explicit_geometry'
       AND completed = 1
       AND V < 0
       AND t0 >= 1.2
       AND corr_dwave > corr_sdw
      THEN 1 ELSE 0
    END
  ) AS frustrated_pair_plateau_runs
FROM runs
;

-- dataset: status_by_selection
SELECT
  CASE selection
    WHEN 'no_explicit_geometry' THEN 'No option (legacy)'
    WHEN 'cubic_unfrustrated' THEN 'Cubic unfrustrated'
    ELSE 'Square'
  END AS selection_label,
  SUM(completed) AS completed,
  SUM(1 - completed) AS checkpoint_only
FROM runs
GROUP BY selection
ORDER BY CASE selection
  WHEN 'no_explicit_geometry' THEN 1
  WHEN 'cubic_unfrustrated' THEN 2
  ELSE 3
END
;

-- dataset: legacy_phase_map
SELECT
  printf('%.1f', t0) AS t0_label,
  t0,
  V,
  log10((corr_dwave + 1.0e-12) / (corr_sdw + 1.0e-12)) AS log10_dwave_to_sdw,
  CASE WHEN corr_dwave > corr_sdw THEN 'd-wave dominant' ELSE 'SDW dominant' END AS dominant_order,
  corr_dwave,
  corr_sdw
FROM runs
WHERE selection = 'no_explicit_geometry' AND completed = 1
ORDER BY V, t0
;

-- dataset: v0_sdw_by_geometry
SELECT
  printf('%.1f', t0) AS t0_label,
  t0,
  MAX(CASE WHEN selection = 'no_explicit_geometry' THEN corr_sdw END) AS cubic_frustrated,
  MAX(CASE WHEN selection = 'cubic_unfrustrated' THEN corr_sdw END) AS cubic_unfrustrated,
  MAX(CASE WHEN selection = 'square' THEN corr_sdw END) AS square
FROM runs
WHERE V = 0 AND completed = 1
GROUP BY t0
ORDER BY t0
;

-- dataset: pair_decay
SELECT
  distance,
  MAX(
    CASE WHEN run_id = 'no_explicit_geometry__V_-0.2__t0_1.4'
      THEN log10(mean_abs) END
  ) AS frustrated_sc_vminus02_t014,
  MAX(
    CASE WHEN run_id = 'square__V_+0.0__t0_1.4'
      THEN log10(mean_abs) END
  ) AS square_v0_t014,
  MAX(
    CASE WHEN run_id = 'cubic_unfrustrated__V_+0.0__t0_1.4'
      THEN log10(mean_abs) END
  ) AS cubic_unfrustrated_v0_t014
FROM pair_correlations
WHERE distance BETWEEN 1 AND 24
GROUP BY distance
ORDER BY distance
;

-- dataset: geometry_comparison
WITH cubic AS (
  SELECT * FROM runs
  WHERE selection = 'cubic_unfrustrated' AND V = 0 AND completed = 1
),
square_runs AS (
  SELECT * FROM runs
  WHERE selection = 'square' AND V = 0 AND completed = 1
)
SELECT
  cubic.t0,
  cubic.corr_sdw AS cubic_sdw,
  square_runs.corr_sdw AS square_sdw,
  100.0 * (1.0 - square_runs.corr_sdw / cubic.corr_sdw) AS square_sdw_reduction_pct,
  cubic.pair_r4 AS cubic_pair_r4,
  square_runs.pair_r4 AS square_pair_r4,
  square_runs.pair_r4 / cubic.pair_r4 AS square_to_cubic_pair_r4_ratio,
  cubic.pair_xi AS cubic_pair_xi,
  square_runs.pair_xi AS square_pair_xi
FROM cubic
JOIN square_runs USING (t0)
ORDER BY cubic.t0
;

-- dataset: pair_summary
SELECT
  CASE run_id
    WHEN 'no_explicit_geometry__V_-0.2__t0_1.4' THEN 'Cubic frustrated, V=-0.2'
    WHEN 'square__V_+0.0__t0_1.4' THEN 'Square, V=0'
    ELSE 'Cubic unfrustrated, V=0'
  END AS case_label,
  V,
  t0,
  pair_r4,
  pair_r8,
  pair_r24,
  pair_xi,
  dwave_order
FROM runs
WHERE run_id IN (
  'no_explicit_geometry__V_-0.2__t0_1.4',
  'square__V_+0.0__t0_1.4',
  'cubic_unfrustrated__V_+0.0__t0_1.4'
)
ORDER BY CASE run_id
  WHEN 'no_explicit_geometry__V_-0.2__t0_1.4' THEN 1
  WHEN 'square__V_+0.0__t0_1.4' THEN 2
  ELSE 3
END
;

-- dataset: perturbative_hierarchy_flags
SELECT
  run_id,
  CASE selection
    WHEN 'no_explicit_geometry' THEN 'Legacy / cubic frustrated'
    WHEN 'cubic_unfrustrated' THEN 'Cubic unfrustrated'
    ELSE 'Square'
  END AS geometry,
  V,
  t0,
  local_ep,
  tp,
  tp_over_ep
FROM runs
WHERE tp_over_ep >= 1.0
ORDER BY geometry, V, t0
;

-- dataset: all_runs
SELECT
  run_id,
  CASE selection
    WHEN 'no_explicit_geometry' THEN 'Legacy / cubic frustrated'
    WHEN 'cubic_unfrustrated' THEN 'Cubic unfrustrated'
    ELSE 'Square'
  END AS geometry,
  V,
  t0,
  CASE
    WHEN completed = 0 THEN 'Checkpoint only'
    WHEN selection = 'no_explicit_geometry' AND V = 0 AND t0 IN (0.8, 1.0)
      THEN 'Complete; recurrent/stale state'
    WHEN alpha_below_floor = 1 THEN 'Complete; pairing below 5e-3 floor'
    ELSE 'Complete'
  END AS review_status,
  CASE dominant
    WHEN 'dwave' THEN 'd-wave'
    WHEN 'sdw' THEN 'SDW'
    ELSE dominant
  END AS dominant,
  corr_sdw,
  corr_dwave,
  corr_cdw,
  sdw_abs_kx_over_pi,
  sdw_ky_over_pi,
  pair_r8,
  gap,
  iterations,
  density_error,
  tp_over_ep
FROM runs
ORDER BY CASE selection
  WHEN 'no_explicit_geometry' THEN 1
  WHEN 'cubic_unfrustrated' THEN 2
  ELSE 3
END, V, t0
;

-- dataset: publication_assessment
SELECT
  finding,
  evidence,
  current_status,
  minimum_next_check
FROM publication_assessment
ORDER BY priority
;
