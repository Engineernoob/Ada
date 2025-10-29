$(document).ready(function () {

    // Initialize text animations for any elements with .text class
    $('.text').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "bounceIn",
        },
        out: {
            effect: "bounceOut",
        },
    });

    // SiriWave initialization
    if (typeof SiriWave !== "undefined") {
        console.log("SiriWave library loaded");
        
        var siriWave = new SiriWave({
            container: document.getElementById("siri-wave-container"),
            width: 800,
            height: 200,
            style: "ios9",
            amplitude: 1,
            speed: 0.30,
            autostart: true
        });

        console.log("SiriWave initialized successfully");

        // Enhanced Siri message system
        var messages = [
            "Hello, I am A.D.A",
            "How can I help you today?",
            "I'm here to assist you",
            "Feel free to ask me anything",
            "I'm excited to chat with you!"
        ];
        
        var currentMessageIndex = 0;
        var isTyping = false;
        
        function showMessage(index, callback) {
            var $message = $("#siri-message");
            var message = messages[index];
            
            // Add "thinking" effect
            siriWave.amplitude = 1.5;
            siriWave.speed = 0.3;
            
            setTimeout(function() {
                // Change to typing indicator
                $message.addClass("typing").text("...");
                
                setTimeout(function() {
                    // Show actual message
                    $message.removeClass("typing").addClass("active").text(message);
                    
                    // Reset wave after message appears
                    siriWave.amplitude = 1;
                    siriWave.speed = 0.3;
                    
                    // Call callback after display time
                    if (callback) {
                        setTimeout(callback, 4000);
                    }
                }, 2000);
            }, 1000);
        }
        
        function messageCycle() {
            showMessage(currentMessageIndex, function() {
                currentMessageIndex = (currentMessageIndex + 1) % messages.length;
                messageCycle();
            });
        }
        
        // Start message cycle after initial delay
        setTimeout(function() {
            showMessage(0, function() {
                currentMessageIndex = 1;
                messageCycle();
            });
        }, 2000);
    }

    // Siri message animation for any elements with .siri-message class
    $('.siri-message').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "fadeInUp",
            sync: true,
        },
        out: {
            effect: "fadeOutUp",
            sync: true,
        },
    });

    // mic button click event
    $("#MicBtn").click(function () {
        // Placeholder for assistant sound
        console.log("Playing assistant sound...");
        
        // Toggle between Oval and SiriWave sections
        $("#Oval").attr("hidden", true);
        $("#SiriWave").attr("hidden", false);
    });

    // Keyboard shortcut for Command+J
    function doc_keyUp(e) {
        if (e.key === 'j' && e.metaKey) {
            console.log("Command+J pressed - activating assistant...");
            $("#Oval").attr("hidden", true);
            $("#SiriWave").attr("hidden", false);
        }
    }
    document.addEventListener('keyup', doc_keyUp, false);

    // Function to play assistant
    function PlayAssistant(message) {
        if (message != "") {
            $("#Oval").attr("hidden", true);
            $("#SiriWave").attr("hidden", false);
            
            // Log the command for now (would normally send to backend)
            console.log("Command received: " + message);
            
            $(".input-field").val("");
            $("#MicBtn").attr('hidden', false);
            $("#Chat").attr('hidden', true);
        }
    }

    // Toggle function to hide and display mic and send button
    function ShowHideButton(message) {
        if (message.length == 0) {
            $("#MicBtn").attr('hidden', false);
            $("#Chat").attr('hidden', true);
        } else {
            $("#MicBtn").attr('hidden', true);
            $("#Chat").attr('hidden', false);
        }
    }

    // Key up event handler on text box
    $(".input-field").keyup(function () {
        let message = $(this).val();
        ShowHideButton(message);
    });

    // Send button event handler
    $("#Chat").click(function () {
        let message = $(".input-field").val();
        PlayAssistant(message);
    });

    // Enter press event handler on chat box
    $(".input-field").keypress(function (e) {
        let key = e.which;
        if (key == 13) {
            let message = $(this).val();
            PlayAssistant(message);
        }
    });

});
