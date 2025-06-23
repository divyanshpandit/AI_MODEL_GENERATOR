// Dark/Light Toggle
const toggle = document.getElementById('theme-toggle');
if (localStorage.theme === 'dark') {
  document.body.classList.add('dark');
  toggle.textContent = '☀️';
}
toggle.addEventListener('click', () => {
  document.body.classList.toggle('dark');
  if (document.body.classList.contains('dark')) {
    localStorage.theme = 'dark';
    toggle.textContent = '☀️';
  } else {
    localStorage.theme = 'light';
    toggle.textContent = '🌙';
  }
});

// SVG Logo Animation
document.querySelector('.logo').addEventListener('mouseover', e => {
  e.target.style.transform = 'rotate(20deg) scale(1.1)';
  e.target.style.transition = 'transform 0.3s';
});
document.querySelector('.logo').addEventListener('mouseout', e => {
  e.target.style.transform = 'rotate(0) scale(1)';
});
