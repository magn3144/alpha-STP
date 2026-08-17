# Notes for CODEX

- Write implementations as simple as possible, reusing exsisting functionality and avoiding overly complicated structures. If you can remove more code than you write as you complete a task, thats great.
- When you want to use py_compile, dont place the cache files in this repo.
- Always place imports at the top of files. Not inside functions.
- Backward compatability doesnt matter, it just complicates the code unnecessarily.
- Dont make fallbacks. Its better to assume that the input to a function always is in the correct format, and just let the code fail if thats not the case.
- After writing code fix any potential Pylance errors.
- If CLI Pyright misses VS Code Pylance errors, inspect the VS Code Python Language Server log for interpreter and editable-install resolution issues.
- Dont write tests. Its not necessary for this project.