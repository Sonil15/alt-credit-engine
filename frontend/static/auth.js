/* Borrower auth helper (token storage + auth-aware fetch.
 * The bearer token is minted by /auth/login|register and kept in localStorage.
 * Everything is local to the browser; no third-party auth service is involved.
 */
(function (global) {
  const TOKEN_KEY = 'altcredit_token';
  const LOGIN_KEY = 'altcredit_login_id';
  const UID_KEY = 'altcredit_user_id';

  function getToken() {
    return localStorage.getItem(TOKEN_KEY) || null;
  }

  function currentUser() {
    const token = getToken();
    if (!token) return null;
    return {
      token,
      login_id: localStorage.getItem(LOGIN_KEY) || '',
      user_id: localStorage.getItem(UID_KEY) || '',
    };
  }

  function isLoggedIn() {
    return !!getToken();
  }

  function _store(data) {
    localStorage.setItem(TOKEN_KEY, data.token);
    localStorage.setItem(LOGIN_KEY, data.login_id);
    localStorage.setItem(UID_KEY, data.user_id);
  }

  function _clearLocal() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(LOGIN_KEY);
    localStorage.removeItem(UID_KEY);
  }

  function authHeaders(extra) {
    const headers = Object.assign({}, extra || {});
    const token = getToken();
    if (token) headers['Authorization'] = 'Bearer ' + token;
    return headers;
  }

  async function _post(path, body) {
    const resp = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || 'Request failed.');
    }
    return data;
  }

  async function register(loginId, password, cibilScore, captchaAnswer, captchaToken) {
    const payload = { login_id: loginId, password };
    if (cibilScore !== undefined && cibilScore !== null) {
      payload.cibil_score = cibilScore;
    }
    if (captchaAnswer !== undefined && captchaAnswer !== null) {
      payload.captcha_answer = captchaAnswer;
    }
    if (captchaToken !== undefined && captchaToken !== null) {
      payload.captcha_token = captchaToken;
    }
    const data = await _post('/auth/register', payload);
    _store(data);
    return data;
  }

  async function login(loginId, password) {
    const data = await _post('/auth/login', { login_id: loginId, password });
    _store(data);
    return data;
  }

  async function logout() {
    const token = getToken();
    if (token) {
      try {
        await fetch('/auth/logout', { method: 'POST', headers: authHeaders() });
      } catch (_) {
        /* best-effort server-side revoke; clear locally regardless */
      }
    }
    _clearLocal();
  }

  global.Auth = {
    getToken,
    currentUser,
    isLoggedIn,
    authHeaders,
    register,
    login,
    logout,
  };
})(window);
