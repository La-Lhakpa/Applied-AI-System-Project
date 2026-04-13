# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

---TriadTune 1.0 

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

--- It suggests a short ranked list from a small fixed song catalog.Each pick is chosen by adding points when the song’s genre and mood tags match what the user chose, and when the song’s energy number is close to the user’s target energy (with energy counting extra in your current setup).
--It assumes the user can describe taste with one genre, one mood, and one energy level, and that those labels mean the same thing as the words stored in the song table. It also assumes energy is on a 0–1 scale and that closer is always better, even if someone might prefer “a little calmer” in real life.
--This is for classroom exploration and reflection, not a production app. The catalog is tiny, the rules are simple on purpose. 

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---After every song has a total score, the system sorts from highest to lowest and shows the top few as recommendations. Songs can still rank fairly high if they only match well on energy, or on genre + energy without the “right” mood—because the rules add the pieces together instead of requiring all three to match perfectly.

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

---There are currently 20 songs in the catalog.
---Genres include pop, indie pop, lofi, rock, metal, punk, jazz, classical, ambient, synthwave, EDM, hip hop, country, folk, reggae, Latin, soul, and a few others—a small mix 
---I added new data with the old data
---Yes. There is no listening history, no language or lyrics, no artist diversity at scale, and no fine-grained subgenres or regional styles

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

During the weight experiment, I saw that doubling energy and halving genre changed who ranked near the top: tracks that matched mood and energy could beat ones that matched genre but not mood, so the list was very sensitive to those knobs even though nothing in the data proved one weighting was “right.” I also noticed that exact genre labels matter a lot—“indie pop” does not count as “pop”—so users who describe taste in everyday language can get recommendations that look misaligned even when the songs feel close. Together, that showed me the system is transparent but brittle: small rule changes and strict string matching can swing results without reflecting deeper understanding of music or listeners.

---

## 7. Evaluation  
I tested four made-up listeners against the full catalog (`data/songs.csv`) using the same scoring code each time:
-- **Happy Pop** — wants pop, a happy mood, and medium-high energy (around 0.8), similar to the default run in `main.py`.  
-- **High-Energy Pop** — still pop, but asks for an intense mood and very high energy (around 0.92).  
-- **Chill Lofi** — wants lofi, a chill mood, and low energy (around 0.38).  
-- **Deep Intense Rock** — wants rock, an intense mood, and high energy (around 0.91).  
  For each profile I printed the **top five** songs and read the built-in reasons (genre match, mood match, energy similarity). I also compared pairs of profiles side by side to see whether the list **actually changed** when the preferences changed, and whether that change matched what the numbers were supposed to reward.

--What surprised me: Gym Hero can rank high for “Happy Pop” just from pop + energy without a happy mood match, and intense high-energy profiles overlapped a lot; Chill Lofi did not surprise me.



---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
