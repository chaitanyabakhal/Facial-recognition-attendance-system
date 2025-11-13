// Main JavaScript for Facial Recognition Attendance System
document.addEventListener('DOMContentLoaded', function() {
  console.log('Facial Recognition Attendance System initialized');

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Auto-hide alerts after 5 seconds
  const alerts = document.querySelectorAll('.alert');
  if (alerts.length > 0) {
    alerts.forEach(alert => {
      setTimeout(() => {
        alert.style.opacity = '0';
        setTimeout(() => { alert.style.display = 'none'; }, 500);
      }, 5000);
    });
  }
});
