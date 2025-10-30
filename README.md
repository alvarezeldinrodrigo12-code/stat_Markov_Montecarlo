# stat_Markov_Montecarlo

# 🔢 From Poetry to Physics: The Origins of Markov Chains and the Monte Carlo Method

## 📘 Overview
This project explores the fascinating journey from **Andrey Markov’s** study of letter sequences in poetry to the **Monte Carlo simulations** that powered the **Manhattan Project**.  
It connects the historical, mathematical, and computational threads that shaped two of the most powerful ideas in modern probability theory.

---

## 🎯 Objectives
- Understand the **historical context** of Markov’s 1913 paper on letter dependencies in Pushkin’s *Eugene Onegin*.
- Demonstrate how Markov Chains laid the foundation for **stochastic processes**.
- Explain how the **Monte Carlo Method** emerged during the 1940s as a computational extension of stochastic modeling.
- Implement simple **Python simulations** to illustrate both concepts.

---

## 🧠 Background

### 🪶 Markov Chains in Poetry
In 1913, Andrey Markov used a 20,000-letter excerpt of Pushkin’s *Eugene Onegin* to study how vowels and consonants alternated.  
He showed that **letter occurrence was not independent**, introducing the concept of **chains of dependent events** — now known as **Markov Chains**.

### ⚛️ Monte Carlo and the Manhattan Project
Three decades later, **Stanislaw Ulam** and **John von Neumann** used random sampling techniques to model neutron diffusion in nuclear reactions.  
These methods became known as the **Monte Carlo Method**, inspired by the idea of **repeated random processes** — directly connected to Markov’s stochastic framework.

---

## 💻 Implementation
The notebook (`markov_montecarlo.ipynb`) contains:

1. **Text Analysis** — Reproducing Markov’s original vowel/consonant experiment using a Russian poem (or English equivalent).  
2. **Transition Matrix Construction** — Building a simple Markov Chain to model text patterns.  
3. **Monte Carlo Simulation** — Estimating π and modeling particle behavior using random sampling.  
4. **Visualization** — Comparing deterministic vs stochastic modeling approaches.

---

## 📊 Example Output
| Experiment | Description | Result |
|-------------|--------------|--------|
| Vowel-Consonant Probability | Frequency of transitions | ~0.53 vowels after consonant |
| Monte Carlo π Estimation | 10⁶ samples | π ≈ 3.1416 |
| Random Walk Simulation | 2D diffusion | Stable average radius over trials |

---

## 🧰 Tools & Libraries
- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- random  

---

## 📚 References
- A. A. Markov (1913), *An Example of Statistical Investigation of the Text Eugene Onegin Illustrating the Dependence Between Successive Letters*.  
- Metropolis, N., & Ulam, S. (1949). *The Monte Carlo Method*. *Journal of the American Statistical Association*, 44(247).  
- Eckmann, J.-P. (2004). *Markov’s Chains and the Birth of Stochastic Processes*.  

---

## 🧩 Project Structure
