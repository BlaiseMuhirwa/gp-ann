format-cpp:
	find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i

format-python:
	find . -name "*.py" | xargs black

format: format-cpp format-python

BUILD_DIRS := release_l2 release_mips

compile-commands:
	@for d in $(BUILD_DIRS); do \
		if [ -d "$$d" ]; then \
			echo ">>> regenerating $$d/compile_commands.json"; \
			cmake -S . -B "$$d" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /dev/null; \
		else \
			echo ">>> skipping $$d (run build.py first)"; \
		fi; \
	done

.PHONY: format-cpp format-python format compile-commands