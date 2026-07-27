## Codex Intructions
- When you want to use py_compile, dont place the cache files in this repo.
- Always place imports at the top of files. Not inside functions.
- Backward compatability doesnt matter, it just complicates the code unnecessarily.
- Dont make fallbacks. Its better to assume that the input to a function always is in the correct format, and just let the code fail if thats not the case.
- Write code as simple as possible, without complicated structures. If you can remove more code than you write as you complete a task, thats great.
- After writing code fix any potential Pylance errors.
- Dont write tests. Its not necessary for this project.
- In function headers write a short description of the function and give a short overview of the inputs and outputs.

## Data Oriented Design
- I want you to use data oriented deign principles for this project.
- That means that functionality and data should be separated.
- Functions should be pure, whenever possible.
- You can use assert in the beginning of a function, but only do this for more complicated functions, where this might be necessary.