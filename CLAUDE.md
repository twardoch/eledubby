# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

`adamdubpy` is a Python tool for voice dubbing that:
- Takes an input video and replaces the audio with a new voice using ElevenLabs API
- Performs speech-to-speech conversion on audio segments
- Splits audio into 10-20 second segments based on silence detection
- Preserves timing by padding or cropping converted segments

## 2. Required Environment Variables

- `ELEVENLABS_API_KEY` - Must be set for the ElevenLabs API authentication (loaded via python-dotenv)

## 3. Key Dependencies

- **elevenlabs-python**: Main package for speech-to-speech conversion
- **python-dotenv**: For loading environment variables
- **ffmpeg**: Required for video/audio processing (must be installed on system)

## 4. Development Commands

### 4.1. Setup
```bash
# Install dependencies
pip install elevenlabs python-dotenv

# Set up environment
echo "ELEVENLABS_API_KEY=your_api_key_here" > .env
```

### 4.2. Running the Tool
```bash
python adamdubpy.py --input video.mp4 --voice voice_id --output output.mp4
```

Default voice ID: `ELEVENLABS_VOICE_ID` environment variable

## 5. Code Architecture

### 5.1. Audio Processing Pipeline
1. **Audio Extraction**: Extract audio from input video
2. **Silence Detection**: Identify split points based on silence (scoring system for silence duration and length)
3. **Segmentation**: Split audio into 10-20 second segments at optimal silence points
4. **Speech-to-Speech**: Convert each segment using ElevenLabs API with specified voice
5. **Timing Preservation**: Pad or crop converted segments to match original timing
6. **Audio Reassembly**: Combine processed segments
7. **Video Remuxing**: Replace original audio track with processed audio

### 5.2. Key Implementation Details
- Segments must be between 10-20 seconds
- Within each 10-20 second window, find the "most silent long pause"
- Score pauses based on both silence level and duration
- Use ElevenLabs speech-to-speech API for voice conversion
- Maintain exact timing alignment with original video

## 6. Important Notes

- The main script file should be named `adamdubpy.py` (if creating from scratch)
- Reference documentation is available in `ref/` directory but contains very large files
- Always check for ffmpeg availability before processing
- Handle API errors gracefully with retry logic
- Preserve original video quality during remuxing



# Software Development Rules

## 7. Pre-Work Preparation

### 7.1. Before Starting Any Work
- **ALWAYS** read `WORK.md` in the main project folder for work progress
- Read `README.md` to understand the project
- STEP BACK and THINK HEAVILY STEP BY STEP about the task
- Consider alternatives and carefully choose the best option
- Check for existing solutions in the codebase before starting

### 7.2. Project Documentation to Maintain
- `README.md` - purpose and functionality
- `CHANGELOG.md` - past change release notes (accumulative)
- `PLAN.md` - detailed future goals, clear plan that discusses specifics
- `TODO.md` - flat simplified itemized `- [ ]`-prefixed representation of `PLAN.md`
- `WORK.md` - work progress updates

## 8. General Coding Principles

### 8.1. Core Development Approach
- Iterate gradually, avoiding major changes
- Focus on minimal viable increments and ship early
- Minimize confirmations and checks
- Preserve existing code/structure unless necessary
- Check often the coherence of the code you're writing with the rest of the code
- Analyze code line-by-line

### 8.2. Code Quality Standards
- Use constants over magic numbers
- Write explanatory docstrings/comments that explain what and WHY
- Explain where and how the code is used/referred to elsewhere
- Handle failures gracefully with retries, fallbacks, user guidance
- Address edge cases, validate assumptions, catch errors early
- Let the computer do the work, minimize user decisions
- Reduce cognitive load, beautify code
- Modularize repeated logic into concise, single-purpose functions
- Favor flat over nested structures

## 9. Tool Usage (When Available)

### 9.1. Additional Tools
- If we need a new Python project, run `curl -LsSf https://astral.sh/uv/install.sh | sh; uv venv --python 3.12; uv init; uv add fire rich; uv sync`
- Use `tree` CLI app if available to verify file locations
- Check existing code with `.venv` folder to scan and consult dependency source code
- Run `DIR="."; uvx codetoprompt --compress --output "$DIR/llms.txt"  --respect-gitignore --cxml --exclude "*.svg,.specstory,*.md,*.txt,ref,testdata,*.lock,*.svg" "$DIR"` to get a condensed snapshot of the codebase into `llms.txt`

## 10. File Management

### 10.1. File Path Tracking
- **MANDATORY**: In every source file, maintain a `this_file` record showing the path relative to project root
- Place `this_file` record near the top:
- As a comment after shebangs in code files
- In YAML frontmatter for Markdown files
- Update paths when moving files
- Omit leading `./`
- Check `this_file` to confirm you're editing the right file

## 11. Python-Specific Guidelines

### 11.1. PEP Standards
- PEP 8: Use consistent formatting and naming, clear descriptive names
- PEP 20: Keep code simple and explicit, prioritize readability over cleverness
- PEP 257: Write clear, imperative docstrings
- Use type hints in their simplest form (list, dict, | for unions)

### 11.2. Modern Python Practices
- Use f-strings and structural pattern matching where appropriate
- Write modern code with `pathlib`
- ALWAYS add "verbose" mode loguru-based logging & debug-log
- Use `uv add` 
- Use `uv pip install` instead of `pip install`
- Prefix Python CLI tools with `python -m` (e.g., `python -m pytest`)

### 11.3. CLI Scripts Setup
For CLI Python scripts, use `fire` & `rich`, and start with:
```python
#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["PKG1", "PKG2"]
# ///
# this_file: PATH_TO_CURRENT_FILE
```

### 11.4. Post-Edit Python Commands
```bash
fd -e py -x uvx autoflake -i {}; fd -e py -x uvx pyupgrade --py312-plus {}; fd -e py -x uvx ruff check --output-format=github --fix --unsafe-fixes {}; fd -e py -x uvx ruff format --respect-gitignore --target-version py312 {}; python -m pytest;
```

## 12. Post-Work Activities

### 12.1. Critical Reflection
- After completing a step, say "Wait, but" and do additional careful critical reasoning
- Go back, think & reflect, revise & improve what you've done
- Don't invent functionality freely
- Stick to the goal of "minimal viable next version"

### 12.2. Documentation Updates
- Update `WORK.md` with what you've done and what needs to be done next
- Document all changes in `CHANGELOG.md`
- Update `TODO.md` and `PLAN.md` accordingly

## 13. Work Methodology

### 13.1. Virtual Team Approach
Be creative, diligent, critical, relentless & funny! Lead two experts:
- **"Ideot"** - for creative, unorthodox ideas
- **"Critin"** - to critique flawed thinking and moderate for balanced discussions

Collaborate step-by-step, sharing thoughts and adapting. If errors are found, step back and focus on accuracy and progress.

### 13.2. Continuous Work Mode
- Treat all items in `PLAN.md` and `TODO.md` as one huge TASK
- Work on implementing the next item
- Review, reflect, refine, revise your implementation
- Periodically check off completed issues
- Continue to the next item without interruption

## 14. Special Commands

### 14.1. `/plan` Command - Transform Requirements into Detailed Plans

When I say "/plan [requirement]", you must:

1. **DECONSTRUCT** the requirement:
- Extract core intent, key features, and objectives
- Identify technical requirements and constraints
- Map what's explicitly stated vs. what's implied
- Determine success criteria

2. **DIAGNOSE** the project needs:
- Audit for missing specifications
- Check technical feasibility
- Assess complexity and dependencies
- Identify potential challenges

3. **RESEARCH** additional material: 
- Repeatedly call the `perplexity_ask` and request up-to-date information or additional remote context
- Repeatedly call the `context7` tool and request up-to-date software package documentation
- Repeatedly call the `codex` tool and request additional reasoning, summarization of files and second opinion

4. **DEVELOP** the plan structure:
- Break down into logical phases/milestones
- Create hierarchical task decomposition
- Assign priorities and dependencies
- Add implementation details and technical specs
- Include edge cases and error handling
- Define testing and validation steps

5. **DELIVER** to `PLAN.md`:
- Write a comprehensive, detailed plan with:
 - Project overview and objectives
 - Technical architecture decisions
 - Phase-by-phase breakdown
 - Specific implementation steps
 - Testing and validation criteria
 - Future considerations
- Simultaneously create/update `TODO.md` with the flat itemized `- [ ]` representation

**Plan Optimization Techniques:**
- **Task Decomposition:** Break complex requirements into atomic, actionable tasks
- **Dependency Mapping:** Identify and document task dependencies
- **Risk Assessment:** Include potential blockers and mitigation strategies
- **Progressive Enhancement:** Start with MVP, then layer improvements
- **Technical Specifications:** Include specific technologies, patterns, and approaches

### 14.2. `/report` Command

1. Read all `./TODO.md` and `./PLAN.md` files
2. Analyze recent changes
3. Document all changes in `./CHANGELOG.md`
4. Remove completed items from `./TODO.md` and `./PLAN.md`
5. Ensure `./PLAN.md` contains detailed, clear plans with specifics
6. Ensure `./TODO.md` is a flat simplified itemized representation

### 14.3. `/work` Command

1. Read all `./TODO.md` and `./PLAN.md` files and reflect
2. Write down the immediate items in this iteration into `./WORK.md`
3. Work on these items
4. Think, contemplate, research, reflect, refine, revise
5. Be careful, curious, vigilant, energetic
6. Verify your changes and think aloud
7. Consult, research, reflect
8. Periodically remove completed items from `./WORK.md`
9. Tick off completed items from `./TODO.md` and `./PLAN.md`
10. Update `./WORK.md` with improvement tasks
11. Execute `/report`
12. Continue to the next item

## 15. Additional Guidelines

- Ask before extending/refactoring existing code that may add complexity or break things
- Work tirelessly without constant updates when in continuous work mode
- Only notify when you've completed all `PLAN.md` and `TODO.md` items

## 16. Command Summary

- `/plan [requirement]` - Transform vague requirements into detailed `PLAN.md` and `TODO.md`
- `/report` - Update documentation and clean up completed tasks
- `/work` - Enter continuous work mode to implement plans
- You may use these commands autonomously when appropriate

