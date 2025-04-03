# AI-Powered Reinforcement Learning for Language Translation in Tourism  

## Overview  

This project explores the application of **Reinforcement Learning (RL)** in developing a robust **language detection and translation model** for **Kinyarwanda-English** translation. Rwanda, a growing tourism destination, faces **significant language barriers** that limit communication between visitors and locals. Existing translation tools often perform poorly with Kinyarwanda due to **limited datasets and lack of contextual understanding**.  

By integrating **Deep Q-Network (DQN)** and **Policy Gradient (PPO/REINFORCE)**, this research investigates how RL can optimize translation accuracy and enhance the user experience. The model is evaluated using **NLP metrics, reinforcement learning benchmarks, and user feedback integration** to measure effectiveness.  

---

## Research Motivation  

### **Background**  
Tourism is a key driver of economic growth in Rwanda. However, **language barriers hinder tourist engagement**, making navigation, cultural interactions, and local business communication challenging. AI-driven translation technologies have transformed language processing, but **low-resource languages** like Kinyarwanda remain underrepresented.  

### **Problem Statement**  
Current translation tools fail to **accurately translate Kinyarwanda** due to:  
- **Limited datasets**  
- **Lack of cultural context**  
- **No user feedback mechanisms**  

This project addresses these challenges using **RL-based optimization techniques**, ensuring real-time translation improvements.  

### **Objectives**  
- Develop an **AI-powered model** to detect and translate Kinyarwanda-English.  
- Optimize translation accuracy using **reinforcement learning techniques**.  
- Integrate **user feedback loops** to refine translations over time.  
- Evaluate performance using **NLP and RL benchmarks**.  

### **Significance**  
This project contributes to:  
- **Enhancing Rwanda’s tourism experience** through better communication.  
- **Empowering local businesses** by improving multilingual interactions.  
- **Advancing AI-driven translation** for low-resource languages.  

---

## Custom Reinforcement Learning Environment  

### **State Space**  
A **continuous 4D vector** representing linguistic and contextual data from text inputs.  

### **Action Space**  
A **discrete action space** where the agent selects translation strategies:  
- **Action 0**: Literal translation  
- **Action 1**: Context-aware transformation  
- **Action 2**: User-adaptive correction  

### **Reward Structure**  
The agent receives rewards based on:  
- **+1** for correct translations (verified by BLEU score).  
- **-1** for incorrect translations (poor semantic coherence).  
- **+0.5** for user-corrected feedback.  

---

## Implemented Methods  

### **1. Deep Q-Network (DQN)**  
- Uses **experience replay** for stable learning.  
- Implements a **target network** to reduce overestimation errors.  
- **Epsilon-greedy strategy** balances exploration vs. exploitation.  

### **2. Policy Gradient (PPO/REINFORCE)**  
- Learns a **stochastic policy** for adaptive translation choices.  
- **Entropy regularization** prevents premature convergence.  
- **Discounted rewards** optimize long-term translation accuracy.  

---

## Performance & Evaluation  

### **1. Agent Performance & Exploration/Exploitation**  
- **DQN excels in structured translation patterns.**  
- **PPO provides more flexibility** but requires more training time.  
- **Balancing exploration vs. exploitation is critical** for improving translations.  

### **2. Simulation Visualization**  
- **Training progress is logged and visualized** with real-time evaluation metrics.  
- **Plots showcase translation improvements** over training episodes.  

### **3. Stable Baselines & Policy Gradient Analysis**  
- **DQN is prone to overfitting** on specific translations.  
- **PPO generalizes better** but needs extensive hyperparameter tuning.  
- **Fine-tuning learning rate and discount factor** improves both models.  

---

## Hyperparameter Optimization  

### **DQN Hyperparameters**  
| Hyperparameter | Optimal Value | Summary |
|---------------|--------------|---------|
| Learning Rate | 0.001 | Maintains stable updates. |
| Gamma (γ) | 0.99 | Ensures long-term reward optimization. |
| Replay Buffer | 10,000 | Enhances learning from past experiences. |
| Batch Size | 32 | Balances memory efficiency and learning speed. |
| Exploration Strategy | Epsilon-Greedy | Reduces overfitting while exploring alternatives. |

### **PPO/REINFORCE Hyperparameters**  
| Hyperparameter | Optimal Value | Summary |
|---------------|--------------|---------|
| Learning Rate | 0.0003 | Ensures gradual policy updates. |
| Gamma (γ) | 0.98 | Rewards long-term translation accuracy. |
| Entropy Coefficient | 0.01 | Prevents policy collapse. |

---

## Running the Project  

### **1. Install Dependencies**  
```bash
pip install -r requirements.txt
## 2. Train Models

### Train DQN
```bash
python train_dqn.py
### **Train PPO/REINFORCE**
python dqn_training.py
python pg_training.py


# Project Structure
Mariam_Azeez_rl_summative/
│
├── environment/
│ ├── custom_env.py # Implementation of custom Gym environment
│ ├── main.py # Main training script entry point
│ ├── rendering.py # Environment visualization utilities
│ ├── requirements.txt # Python dependencies for environment
│ ├── test_environment.py # Unit tests for environment
│
├── models/
│ ├── ppo_language_translation.model # Pretrained PPO model weights
│ ├── pg_language_translation.model # Pretrained Policy Gradient model weights
│
├── training/
│ ├── dqn_training.py # Deep Q-Network implementation and training
│ ├── pg_training.py # Policy Gradient/PPO implementation and training
│
├── evaluation/
│ ├── evaluate_agent.py # Script for evaluating agent performance
│
├── README.md # Project documentation and instructions



Key improvements made:
1. Added proper markdown code block syntax (triple backticks)
2. Fixed indentation in the directory tree
3. Added missing `.py` extension for `evaluate_agent`
4. Improved section headers for better readability
5. Maintained consistent spacing


---

### **Static File **  
- A simple diagram or image representing the **agent’s state space** and **translation choices**.  
- Helps visualize how RL is used to **optimize language translation accuracy**.  
![rl](https://github.com/user-attachments/assets/c651b3bb-85b9-4dc7-a173-5ea8b7591a27)


---
---
**Author:** Mariam Azeez  
**Project:** AI-Powered Reinforcement Learning for Language Translation in Tourism  
**Date:** April 2025  

