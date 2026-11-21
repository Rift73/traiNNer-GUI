// Runs before first paint (classic script in <head>) so the page never flashes the wrong theme.
(function () {
  var theme = null;
  try {
    theme = JSON.parse(localStorage.getItem('trainner-gui:theme'));
  } catch (e) {
    theme = null;
  }
  if (theme !== 'light' && theme !== 'dark') {
    theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.dataset.theme = theme;
})();
