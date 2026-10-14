-- 고객 감정이 크게 오르기 직전의 상담원 발화 (회복 계기 후보)
SELECT s.source_id AS call_id,
       ROUND(e.delta, 2) AS valence_rise,
       e.start_sec,
       a.text AS agent_turn_before
FROM events e
JOIN sessions s ON s.session_id = e.session_id
LEFT JOIN units a ON a.unit_id = e.trigger_unit_id
WHERE e.index_name = 'valence' AND e.delta > 0
ORDER BY e.delta DESC;
