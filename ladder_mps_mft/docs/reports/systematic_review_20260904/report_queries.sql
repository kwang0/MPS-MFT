SELECT category, COUNT(*) AS count,
       COUNT(*) * 1.0 / (SELECT COUNT(*) FROM review_terminal_artifacts) AS fraction
FROM review_terminal_artifacts
GROUP BY category
ORDER BY CASE category
 WHEN 'Stored accepted; passes screen' THEN 1
 WHEN 'Stored accepted; fails screen' THEN 2
 WHEN 'Unaccepted; history available' THEN 3
 WHEN 'Older history unavailable' THEN 4 ELSE 5 END;

SELECT seed, iterations,
       (corrected_per_site - MIN(corrected_per_site) OVER ()) * 1000000.0
           AS delta_e_micro_t_per_site,
       ABS(mu * (density - target_density)) * 1000000.0
           AS density_correction_micro_t_per_site,
       last_sweep_change, status, model_fingerprint, numerical_fingerprint
FROM review_current_seed_inputs
ORDER BY seed ASC;
