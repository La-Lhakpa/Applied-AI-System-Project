# Model Card: EchoMind 1.0

## 1. Model Name
EchoMind 1.0: Intent-Aware Hybrid Music Recommender**

## 2. Intended Use
EchoMind is designed to produce a ranked top-k list of songs from a fixed catalog for classroom experimentation with transparent recommendation logic.

- **Primary use case:** educational exploration of recommender system design, reliability, and explainability.
- **Target users:** students, instructors, and developers testing recommendation behavior.
- **Assumptions about user input:** the user can provide one genre, one mood, a target energy value (0-1), and an optional short intent phrase.
- **Not intended for production:** this is not a large-scale consumer recommendation product and should not be used for high-stakes personalization.

## 3. How the Model Works
EchoMind combines two scoring signals for each song and then applies an optional policy layer:

1. **Content score:** cosine similarity between the user taste vector and each song feature vector.
2. **Label score:** deterministic matching based on genre, mood, and energy alignment.
3. **Hybrid blend:** final base score is a weighted blend of content and label scores.
4. **Agentic policy adjustments:** intent text and session feedback (likes/skips) can adjust blending, apply boosts/filters, and change target energy.
5. **Guardrail + fallback:** if policy fails or hard filters are too strict, the pipeline falls back or relaxes filters to avoid empty/fragile outputs.
6. **Traceability:** the system records reasoning steps, confidence, metrics, and fallback usage.

Compared with the starter system, this version adds policy-aware reranking, confidence scoring, reasoning trace visibility, and deterministic fallback behavior.

## 4. Data
- **Source:** `data/songs.csv`
- **Catalog size:** 50 songs
- **Coverage:** mixed genres and moods (for example pop, lofi, rock, jazz, EDM, classical, hip hop, country, folk, reggae, Latin, soul).
- **Data updates:** additional songs were added beyond the starter set.
- **Missing dimensions:** no listening history, no lyric semantics, no social/contextual features, limited artist and cultural coverage, and limited long-tail/subgenre representation.

## 5. Strengths
EchoMind works best when user preferences are explicit and aligned with available catalog labels.

- Produces consistent, deterministic rankings for identical inputs.
- Explains results with decomposed scores and rationale text.
- Handles optional intent and session feedback without making the model opaque.
- Includes fallback/guardrail logic so recommendation runs remain usable under failures or strict filters.
- Performs strongly on most automated checks (`22/23` tests passing).

## 6. Limitations and Bias
- **Small-catalog bias:** recommendations reflect a limited dataset and may underrepresent broader musical diversity.
- **Label rigidity:** exact string/tag matching can miss near-synonyms (`indie pop` vs `pop`), which can feel unfair to users with natural-language phrasing.
- **Weight sensitivity:** rankings can shift significantly with parameter/weight changes, revealing brittleness in hand-tuned rules.
- **Rule-based intent parsing:** intent interpretation is deterministic and can fail on edge phrasing.
- **Short-term personalization only:** likes/skips are session-level and do not model long-term behavior.
- **No explicit fairness objective:** artist/genre diversity constraints are not formally enforced in ranking.

## 7. Evaluation
### Automated Reliability Checks
- Test suite result: 22 passed, 1 failed (23 total).
- The single failure is an intent-driven reranking case where `"only chill lofi focus"` did not move the expected lofi song to the top.

### Confidence and Observability
- The policy layer assigns confidence values (high confidence for clear signals, slightly lower for partial signals).
- The UI and trace output expose confidence, policy decisions, and whether fallback was triggered.

### Human Evaluation
I manually reviewed outputs for contrasting listener profiles (for example happy pop, chill lofi, and high-energy rock) and checked whether ranking changes were intuitive given profile changes. In most cases, output shifts matched expectations, while the failed intent edge case highlighted brittleness in strict rule-based interpretation.

### Summary
EchoMind is reliable for baseline ranking and fallback behavior, but intent-driven reranking still has one measurable edge-case reliability gap.

## 8. Potential Misuse and Mitigations
### Potential Misuse
- Users can game intent text (for example using restrictive phrases such as "only ...") to force narrow outputs.
- Dataset edits or mislabeled rows can skew recommendations.
- Users may over-trust output quality because the system appears confident/explainable.

### Mitigations
- Bound policy adjustment magnitudes and monitor hard-filter usage.
- Validate dataset schema/ranges before loading.
- Keep explicit user-facing disclaimers that this is an educational recommender.
- Log failures and fallback events for post-run review.

## 9. Future Work
- Improve intent parsing robustness (synonyms, phrase understanding, softer constraints).
- Add diversity-aware reranking to reduce repeated genre/artist concentration.
- Expand dataset size and coverage for better generalization.
- Introduce longer-horizon user memory beyond a single session.
- Add targeted tests for natural-language intent edge cases to improve reliability.

## 10. Personal Reflection
This project reinforced that trustworthy AI systems are built through transparency, testing, and fallback planning, not only through model complexity. The most useful lesson was seeing that high overall pass rates can still hide important edge-case failures, especially for natural-language control signals.
