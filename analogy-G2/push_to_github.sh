#!/bin/bash
# Script to push analogy-G2 to GitHub

cd /Users/xiongjiaqi/Desktop/OII_SDS/FSDS/Group2/fsds25-analogy/analogy-G2

echo "🔍 Checking git status..."
git status

echo ""
echo "🔗 Checking remote configuration..."
git remote -v

echo ""
echo "🔄 Updating remote URL..."
git remote set-url origin git@github.com:kikixiong/FSDS25_Analogy_G2.git

echo ""
echo "✅ Remote updated. Current configuration:"
git remote -v

echo ""
echo "🌿 Ensuring branch is 'main'..."
git branch -M main

echo ""
echo "📦 Staging all files..."
git add .

echo ""
echo "💾 Committing changes..."
git commit -m "Initial commit: Analogy Testing Platform"

echo ""
echo "🚀 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Repository pushed to GitHub."

