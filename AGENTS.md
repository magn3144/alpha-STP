# Notes for CODEX

- Write implementations as simple as possible, reusing exsisting functionality and avoiding overly complicated structures. If you can remove more code than you write as you complete a task, thats great.
- When you want to use py_compile, dont place the cache files in this repo.
- Always place imports at the top of files. Not inside functions.
- Backward compatability doesnt matter, it just complicates the code unnecessarily.
- Dont make fallbacks. Its better to assume that the input to a function always is in the correct format, and just let the code fail if thats not the case.
- Do not use type annotations.
- Dont write tests. Its not necessary for this project.
- Dont edit the README.md, unless I explicitly ask you to.
- When I ask a question give me a concise answer that gets straight to the point.
- When I simply ask you a question, just answer the question without changing anything.


# Design Choices

- Adapted to run on GPUs instead of TPUs
- In the original code the pass rate to train the conjecturer on a conjecture is 0 < pass < 0.25. I changed it to 0 < pass <= 0.25
- Since we use a much smaller global batch size we used square root scaling to find an appropriate learning rate. Warmup steps were scaled as a fraction of optimizer steps.