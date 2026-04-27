# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works
Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

--- The system will be working on the basis of scoring and ranking rules. 
--- My UserProfile will have info's on favorite_genre,favorite_mood,target_energy,likes_acoustic.
---My Recommender compute a score:weighted sum of (genre match + mood match + energy closeness + acoustic preference), with energy as distance-to-target, and only add tempo/valence/danceability when you have a clear rule or extra profile fields.
---Score every candidate, sort by score descending, return the top k—and optionally post-process for ties, diversity, or exclusions.


## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
CLI: python -m src.main

UI: streamlit run src/app.py
```


Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---Sample terminal output from `python -m src.main` (ranked titles, scores, and scoring reasons):

![Terminal output showing top song recommendations with scores and bulleted reasons](docs/terminal-recommendations.png)

Top 5 recommendations for three example profiles (High-Energy Pop, Chill Lofi, Deep Intense Rock):

![Terminal output showing top five songs per profile with scores](docs/terminal-three-profiles.png)

### Running Tests

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

-Short profile comparisons
Happy Pop ↔ High-Energy Pop — Winner switches from Sunrise City (happy + pop) to Gym Hero (intense + pop); mood label has to match or you lose a point.

Happy Pop ↔ Chill Lofi — Totally different tops: pop/happy/mid energy vs lofi/chill/low energy. Spacewalk Thoughts can appear for lofi users on chill + calm energy without matching genre.

Happy Pop ↔ Deep Intense Rock — Sunrise City-style picks vs Storm Runner first; Gym Hero can rank high for rock or intense pop when intense + high energy line up.

High-Energy Pop ↔ Chill Lofi — Loud intense list vs quiet chill list; goals are opposite so the tops barely overlap.

High-Energy Pop ↔ Deep Intense Rock — Same two leaders (Gym Hero, Storm Runner) but order swaps: genre decides which wins; tail is still “loud” songs getting mostly energy points.

Chill Lofi ↔ Deep Intense Rock — Soft lofi/chill vs hard rock/intense; no blending—one profile at a time.


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---EchoMind 1.0 

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits
---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

---

## Agentic Workflow Integration

EchoMind now includes one integrated AI feature: an **agentic ranking policy** that runs
inside `RecommendationPipeline.run()` (not as a standalone script).

### What the policy uses

- `UserProfile` defaults (genre, mood, target energy, acoustic preference)
- Optional natural-language intent from Streamlit (examples: "calm coding mix", "high-energy workout")
- Optional lightweight session feedback (`likes`, `skips`)

### What the policy changes

- Dynamic `blend_alpha` (content-vs-label weighting)
- Optional hard filters (e.g., "only chill lofi")
- Soft boosts for genre/mood/acousticness
- Energy-proximity bonus around policy-adjusted target energy
- Human-readable policy rationale added to each recommendation explanation

### Deterministic fallback

If no intent/feedback is provided, or if policy computation fails, the pipeline falls back to
the baseline deterministic ranking behavior.

