# ReadyQueue Documentation

This document provides a detailed overview of the `ReadyQueue` component within the `cgadimpl` codebase. It outlines the namespaces, classes, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `ReadyQueue` is a **thread-safe producer-consumer queue** designed to manage the execution order of nodes in the computational graph. It is typically used in parallelized versions of topological sorting or gradient execution, where nodes ("tasks") are pushed to the queue once their dependencies have been satisfied (i.e., they are "ready" to execute).

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.

## Dependencies

The `ReadyQueue` component is header-only (`include/ad/core/ReadyQueue.hpp`) and relies on the following:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<queue>` | Standard Library | Underlying container for storing `Node*`. |
| `<mutex>`, `<condition_variable>` | Standard Library | Primitives for thread safety and blocking operations. |
| `<atomic>` | Standard Library | (Included but not explicitly used in the shown class, likely for future use). |
| `ad/core/graph.hpp` | Internal | Defines the `Node` structure that this queue manages. |

## Class: `readyqueue`

The class `ag::readyqueue` encapsulates the synchronization logic.

### Functions Declared

```cpp
namespace ag {
class readyqueue {
public:
    void push(Node* node);
    Node* pop();
    void shutdown();
};
}
```

### Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`push`** | Adds a `Node*` to the queue securely. <br> 1. Acquires lock. <br> 2. Pushes node. <br> 3. Notifies one waiting thread (via `cv.notify_one`). | `std::lock_guard`, `std::queue::push`, `std::condition_variable` |
| **`pop`** | Retrieval method. <br> 1. Acquires lock. <br> 2. **Blocks** (waits) until the queue is not empty OR `shutdown()` has been called. <br> 3. If shutdown and empty, returns `nullptr`. <br> 4. Otherwise, returns the front `Node*`. | `std::unique_lock`, `std::condition_variable::wait` |
| **`shutdown`** | Signals that no more work will be submitted. <br> 1. Sets `finished = true`. <br> 2. Wakes up **all** waiting threads (`cv.notify_all`) so they can exit gracefully (returning `nullptr`). | `std::lock_guard`, `std::condition_variable` |

## Analysis: Pros and Cons

### Pros
*   **Thread Safety**: fully protects the underlying `std::queue` with a `mutex`, allowing safe concurrent access from multiple worker threads.
*   **Blocking Semantics**: The `pop()` method efficiently sleeps using a condition variable when there is no work, avoiding CPU-intensive spin-waiting.
*   **Graceful Shutdown**: The `shutdown()` mechanism allows the consumer threads to exit loops cleanly by checking for `finished && empty`.

### Cons & Potential Drawbacks
*   **Single-Use Design**: The `finished` flag is set to `true` in `shutdown()` but never reset. Once shutdown, the queue object cannot be reused for a new batch of work unless recreated.
*   **Potential Deadlock (Logic specific)**: If consumers are waiting in `pop()` and the producer crashes or finishes without calling `shutdown()`, the consumers will hang indefinitely. Usage requires strict adherence to calling `shutdown()`.
*   **Memory Leaks (Indirect)**: The queue stores raw `Node*` pointers. It does not own them. If the queue is destroyed while non-empty, the queue itself cleans up the pointers but does *not* delete the Nodes (which is correct, as Nodes are managed by `shared_ptr` in the graph, but care must be taken that Nodes don't vanish while sitting in the queue).
*   **Header-Only**: Implementation is inline. This increases compilation time if included in many translation units, though for a small class it's negligible.
