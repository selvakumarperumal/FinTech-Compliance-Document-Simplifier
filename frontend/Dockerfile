# Base stage
FROM node:20-alpine AS base

WORKDIR /app

# Copy package configuration files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Development stage
FROM base AS development

ENV NODE_ENV=development

EXPOSE 5173

# Run development server
CMD ["npm", "run", "dev"]

# Build stage for production
FROM base AS builder

# Copy frontend source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:1.25-alpine AS production

# Copy built static files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
