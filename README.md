## TODO:

### Algorithm release:
- Add README -> The evolution from [Tree of Thought](https://arxiv.org/abs/2305.10601) to [Improving mathematical reasoning with process-supervision](https://openai.com/research/improving-mathematical-reasoning-with-process-supervision). The transition from 1 iteration to multiple reasoning traces as atomic steps, then the intent of OpenAI of using Human Feedback and latest the Q-Value determination by using backpropagation using ground truth signal.
- Add dependencies (possible poetry to track them).
- Verify the score accumulation logic -> ISSUE: WHEN ACCUMULATING WE CAN GO INTO LOOPS. IT WILL ALWAYS PRIORITIZE THE DEPEST BRANCHES... (Sol: depth penalty / noramlize..)
- Should we save all the evaluations from GPT-4 (for Opensource model) or only the adjusted ones?.
- Automatically run the math dataset? Download it and start training directly.
- Verify performance (pick hard problems, see the branch change.)

---
### Open source model:
- Prepare an implementation to automatically fine-tune an Open-source LLM data. (OpenHermes2.5?) - WIP, model downloaded, TODO: Prepare script.
- Real-time tracking of the performance of the Open-Source model, we can do public collaboration. (Prohibitive license here.)