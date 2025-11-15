function login() {
    const user = document.getElementById('username').value;
    const pass = document.getElementById('password').value;
  
    // Replace with a backend check for real apps
    const VALID_USER = 'admin';
    const VALID_PASS = 'drone2025';
  
    if (user === VALID_USER && pass === VALID_PASS) {
      sessionStorage.setItem('authorized', 'true');
      window.location.href = 'index.html';
    } else {
      document.getElementById('loginError').textContent = 'Invalid credentials!';
    }
  }


  // run after DOM is ready
  document.addEventListener('DOMContentLoaded', function () {
    // call login() when Enter is pressed anywhere on the page
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.keyCode === 13) {
        // prevent default form submission behavior if any
        e.preventDefault();
        // call your existing login function
        login();
      }
    });
  });
