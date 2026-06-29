// Session inactivity timeout logic
(function() {
  // 15 minutes in milliseconds
  const DEFAULT_TIMEOUT = 15 * 60 * 1000;
  
  // Allow overriding timeout duration via localStorage (useful for debugging/testing, e.g., set to 5000 for 5s)
  const TIMEOUT_DURATION = parseInt(localStorage.getItem('SESSION_TIMEOUT_DURATION')) || DEFAULT_TIMEOUT;
  
  let timeoutTimer;

  function resetTimer() {
    clearTimeout(timeoutTimer);
    timeoutTimer = setTimeout(logout, TIMEOUT_DURATION);
  }

  function logout() {
    // Clear session storage if any session data exists
    sessionStorage.clear();
    // Redirect to landing page with a timeout parameter
    window.location.href = '/?timeout=true';
  }

  // Events that represent user activity
  const activityEvents = ['mousemove', 'keydown', 'mousedown', 'touchstart', 'scroll', 'click'];

  // Attach event listeners
  activityEvents.forEach(eventName => {
    document.addEventListener(eventName, resetTimer, true);
  });

  // Start the initial timer
  resetTimer();
})();
