# Stage 1: Build the React application using Node.js v22
FROM node:22-alpine AS build

WORKDIR /app

# Copy package.json and lock file
COPY package*.json ./

# Install dependencies
# If you have a package-lock.json or yarn.lock, consider `npm ci` or `yarn install --frozen-lockfile`
RUN npm install

# Copy the rest of the application source code
COPY . .

# Vite uses environment variables prefixed with VITE_
# We will handle API_BASE_URL at runtime in Nginx, so no build-time ARG needed here for that.
# However, if you had other build-time VITE_ variables, you could set them:
# ARG VITE_SOME_OTHER_VAR
# ENV VITE_SOME_OTHER_VAR=${VITE_SOME_OTHER_VAR}

# Build the application for production
RUN npm run build
# The build output will be in /app/dist

# Stage 2: Serve the application with Nginx
FROM nginx:1.25-alpine

# Copy the build output from the 'build' stage to Nginx's web root
COPY --from=build /app/dist /usr/share/nginx/html

# Remove default Nginx configuration (optional, but good practice if replacing entirely)
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the shell script to substitute environment variables at runtime
COPY substitute-env.sh /docker-entrypoint.d/20-substitute-env.sh
RUN chmod +x /docker-entrypoint.d/20-substitute-env.sh

# Expose port 80 (Nginx default)
EXPOSE 80

# Nginx entrypoint will automatically run scripts in /docker-entrypoint.d/
# Then Nginx starts via CMD ["nginx", "-g", "daemon off;"] from its base image.