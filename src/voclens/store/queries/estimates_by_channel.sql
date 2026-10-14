-- 채널별 추정 CSAT·CES·NPS 성향 (voclens cx 실행 후)
SELECT s.source_type AS channel,
       COUNT(*) AS n_sessions,
       ROUND(AVG(e.csat_est), 2) AS csat_est,
       ROUND(100.0 * AVG(e.csat_est >= 4), 1) AS csat_pct,
       ROUND(AVG(e.ces_est), 2) AS ces_est,
       SUM(e.nps_group = 'promoter') AS promoters,
       SUM(e.nps_group = 'detractor') AS detractors,
       SUM(e.nps_group = 'unknown') AS unknown
FROM estimates e
JOIN sessions s ON s.session_id = e.session_id
GROUP BY s.source_type
ORDER BY n_sessions DESC;
