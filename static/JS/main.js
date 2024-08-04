document.addEventListener('DOMContentLoaded', function() {
  const dropdowns = document.querySelectorAll('.dropdown-wrapper');

  dropdowns.forEach(dropdown => {
    const toggleButton = dropdown.previousElementSibling;
    const menu = dropdown.querySelector('.dropdown-menu');

    toggleButton.addEventListener('click', function(event) {
      event.stopPropagation(); // Prevent click event from bubbling up
      menu.classList.toggle('hidden');
    });

    document.addEventListener('click', function(event) {
      if (!dropdown.contains(event.target)) {
        menu.classList.add('hidden');
      }
    });
  });
});