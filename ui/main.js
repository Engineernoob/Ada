$(document).ready(function() {
    // Initialize textillate for Siri message animation
    $(".siri-message").textillate({
        loop: false,
        in: {
            effect: "fadeInUp",
            sync: true
        },
        out: {
            effect: "fadeOutUp",
            sync: true
        }
    });
    
    // Initialize Siri Wave animation
    const SiriWave = new SiriWave({
        container: document.getElementById("siri-wave-container"),
        width: 320,
        height: 100,
        style: "ios9",
        amplitude: 1,
        speed: 0.2,
        autostart: true
    });
});
