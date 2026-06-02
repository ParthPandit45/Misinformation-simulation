const fs = require('fs');
const path = require('path');

const src = path.resolve(__dirname, '..', 'results', 'summary_results.json');
const destDir = path.resolve(__dirname, '..', 'public', 'weights');
const dest = path.join(destDir, 'custom_results.json');

if (!fs.existsSync(src)) {
  console.error('Source summary_results.json not found at', src);
  process.exit(1);
}
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}
fs.copyFileSync(src, dest);
console.log(`Copied ${src} to ${dest}`);
