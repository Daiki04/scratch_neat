from concurrent.futures import ThreadPoolExecutor
import time

def func1():
    for i in range(5):
        print(f"func1-{i}")
        time.sleep(1)
    return "func1 done"

def func2():
    for i in range(5):
        print(f"func2-{i}")
        time.sleep(2)
    return "func2 done"

def func3():
    for i in range(2):
        print(f"func3-{i}")
        time.sleep(3)
    return "func3 done"

def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(lambda f: f(), [func1, func2, func3])

    for result in results:
        print(result)

if __name__ == "__main__":
    main()