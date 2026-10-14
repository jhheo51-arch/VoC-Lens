-- 영상 댓글의 불만 유형 (시청자가 많이 말한 불만)
SELECT s.source_id AS video_id,
       l.label,
       COUNT(*) AS n_comments,
       SUM(CASE WHEN u.start_sec IS NOT NULL THEN 1 ELSE 0 END) AS with_timestamp
FROM unit_labels l
JOIN units u ON u.unit_id = l.unit_id
JOIN sessions s ON s.session_id = u.session_id
WHERE u.source_type = 'video' AND u.speaker_role = 'commenter' AND l.label_type = 'complaint' AND l.category IS NOT NULL
GROUP BY s.source_id, l.label
ORDER BY s.source_id, n_comments DESC;
