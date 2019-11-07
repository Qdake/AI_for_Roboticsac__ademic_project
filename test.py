import multiprocessing
def task(pid):
    # do something
    return pid

def main():
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    results = []
    for i in range(0, cpus):
        result = pool.apply_async(task, args=(i,))
        results.append(result)

    pool.close()
    pool.join()

    for result in results:
        print(result.get())
main()