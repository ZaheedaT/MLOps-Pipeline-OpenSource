const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navLinks");
const year = document.getElementById("year");


// Current year in footer
if (year) {
    year.textContent = new Date().getFullYear();
}


// Mobile navigation
if (navToggle && navLinks) {

    navToggle.addEventListener("click", () => {

        const isOpen =
            navLinks.classList.toggle("open");

        navToggle.setAttribute(
            "aria-expanded",
            String(isOpen)
        );

        document.body.classList.toggle(
            "menu-open",
            isOpen
        );
    });


    // Close mobile navigation after selecting a section
    navLinks
        .querySelectorAll("a")
        .forEach((link) => {

            link.addEventListener("click", () => {

                navLinks.classList.remove("open");

                navToggle.setAttribute(
                    "aria-expanded",
                    "false"
                );

                document.body.classList.remove(
                    "menu-open"
                );

            });

        });
}
