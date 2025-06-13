# Interactive Placement and Reinforcement Loop

The following tasks outline how to integrate an interactive puzzle placement workflow with reinforcement learning. Complete them in order to build and train the system.

## 1. Suggestion API
- [ ] Add a new Flask route `/suggest_match` in `server.py` that accepts a target piece ID and edge index.
- [ ] Use `puzzle.scoring.top_n_matches` to return the best candidate pieces and edges.
- [ ] Return results as JSON with the candidate IDs, edge indices and scores.

## 2. Frontend Controls
- [ ] Update the Next.js UI to allow selecting a piece and one of its edges.
- [ ] Display the ranked suggestions returned by `/suggest_match`.
- [ ] Provide buttons for **Accept**, **Reject** and **Skip** on each suggestion.

## 3. Feedback Capture
- [ ] When the user responds, send the feedback back to the server via a `/submit_feedback` endpoint.
- [ ] Record the `(state, action, reward)` triple for reinforcement learning. Accepted suggestions get a reward of `+1`, rejected `-1`, skipped `0`.
- [ ] Persist feedback to a JSONL or CSV file so it can be batched later.

## 4. Reinforcement Learning Trainer
- [ ] Implement a small gym-like environment that exposes the puzzle state and candidate actions.
- [ ] Use Stable Baselines3 (e.g. PPO or DQN) to train a policy on the collected feedback data in batches.
- [ ] Provide a script `train_rl.py` that loads the feedback file and updates the model weights.

## 5. Model Integration
- [ ] Load the trained model in `server.py` and use it to rank suggestions inside `/suggest_match`.
- [ ] Allow resetting to the previous good model if rankings degrade.
- [ ] Log each suggestion and decision for debugging.

## 6. Usage Instructions
- [ ] Document the workflow in `README.md`:
  1. Start the Flask server with `python server.py`.
  2. Run the Next.js frontend and open it in the browser.
  3. Select a puzzle piece, view suggestions and provide feedback.
  4. Periodically run `python train_rl.py` to update the policy.

Completing these tasks will create a feedback-driven assembly loop that becomes more accurate over time.
