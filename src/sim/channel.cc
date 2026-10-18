#include "gclab/sim/channel.h"

#include <algorithm>
#include <string>
#include <utility>

#include "gclab/common/error.h"

namespace gclab::sim {

void Channel::push(Message msg) {
  // TODO(step 2):
  //   1. Take the lock, append msg to queue_.
  //   2. Release the lock, then notify.
  //   Question to answer in your notes: why is it fine (and slightly better)
  //   to call notify_* AFTER unlocking? Should this be notify_one or
  //   notify_all, given that abort() also uses cv_?
  {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.push_back(std::move(msg));
  }
  // Different waiters may be looking for different tags. Waking only one can
  // wake a waiter that cannot consume this message and leave the matching
  // waiter asleep until its deadline.
  cv_.notify_all();
}

Message Channel::pop(int tag, std::chrono::milliseconds timeout) {
  // TODO(step 2):
  //   1. Compute an absolute deadline ONCE (steady_clock::now() + timeout).
  //      Why not pass `timeout` to every wait call?
  //   2. Under the lock, loop:
  //        - if aborted_  -> throw AbortedError
  //        - scan queue_ for the first message whose tag matches; if found,
  //          move it out, erase it, return it
  //        - otherwise cv_.wait_until(lock, deadline); if that returns
  //          std::cv_status::timeout AND there is still no match, throw
  //          TimeoutError with a message that includes the tag.
  //   3. Spurious wakeups exist. Your loop must tolerate them.
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::unique_lock<std::mutex> lock(mu_);

  for (;;) {
    if (aborted_) throw AbortedError("channel aborted");

    const auto it = std::find_if(queue_.begin(), queue_.end(),
                                 [tag](const Message& msg) { return msg.tag == tag; });
    if (it != queue_.end()) {
      Message msg = std::move(*it);
      queue_.erase(it);
      return msg;
    }

    if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      // Recheck while holding the lock: a push may have raced with the
      // deadline and made the requested message available.
      if (aborted_) throw AbortedError("channel aborted");
      const auto match = std::find_if(queue_.begin(), queue_.end(),
                                      [tag](const Message& msg) { return msg.tag == tag; });
      if (match == queue_.end()) {
        throw TimeoutError("timed out waiting for tag " + std::to_string(tag));
      }
    }
  }
}

void Channel::abort() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    aborted_ = true;
  }
  cv_.notify_all();
}

std::size_t Channel::pending() const {
  std::lock_guard<std::mutex> lock(mu_);
  return queue_.size();
}

}  // namespace gclab::sim
