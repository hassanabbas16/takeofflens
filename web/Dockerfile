FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
# npm ci installs exactly the locked tree and fails if package.json and the lockfile
# disagree, which is what makes the image reproducible. npm install would silently resolve
# a newer tree on every rebuild.
RUN npm ci

COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev"]
