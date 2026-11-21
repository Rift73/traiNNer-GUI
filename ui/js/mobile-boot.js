// Phone edition, before first paint (classic script in <head>): the theme, from this edition's own
// preference or else from the phone's setting, and the colour of the browser's bars to match.
(function () {
  var theme = null;
  try {
    theme = JSON.parse(localStorage.getItem('trainner-gui:m:theme'));
  } catch (e) {
    theme = null;
  }
  if (theme !== 'light' && theme !== 'dark') {
    theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.dataset.theme = theme;
  var bar = document.querySelector('meta[name="theme-color"]');
  if (bar) bar.setAttribute('content', theme === 'dark' ? '#303030' : '#ffffff');
})();
